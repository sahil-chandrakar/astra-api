import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from app.config import Settings
from app.models import (
    AutomationCancelRequest,
    AutomationConfirmRequest,
    AutomationContinueRequest,
    AutomationEvent,
    AutomationRecipe,
    AutomationRecipeCreateRequest,
    AutomationRun,
    AutomationRunRequest,
)
from app.services.llm import LlmService


class AutomationCancelledError(Exception):
    """Raised when the user cancels an active automation run."""


ALLOWED_AUTOMATION_TOOLS = {
    "browser.open",
    "browser.search",
    "browser.click",
    "browser.type",
    "browser.extract",
    "browser.wait_for_user",
    "browser.download",
    "python.run_safe",
    "video.download_permitted",
    "automation.save_recipe",
}

PRIVATE_HOSTS = ("mail.google.com", "gmail.com", "accounts.google.com")
VIDEO_DOWNLOAD_MAX_SIZE_BYTES = 500 * 1024 * 1024
VIDEO_DOWNLOAD_MAX_SIZE_MB = 500
VIDEO_DOWNLOAD_TIMEOUT_SECONDS = 15 * 60
VIDEO_DOWNLOAD_FORMAT_SELECTOR = (
    "bv*[protocol=https]+ba[protocol=https]/"
    "bv*+ba/"
    "best[protocol=https]/"
    "best"
)
VIDEO_DOWNLOAD_FALLBACK_FORMAT_SELECTOR = "18/22/best[protocol=https][vcodec!=none][acodec!=none]/best"
BLOCKED_VIDEO_AVAILABILITY = {
    "private",
    "premium_only",
    "subscriber_only",
    "needs_auth",
    "unlisted",
}
BLOCKED_PROMPT_PATTERN = re.compile(
    r"\b(password|credit card|payment|purchase|buy|delete all|format|wipe|steal|token|cookie|drm|bypass|pirated|torrent|hack)\b",
    re.IGNORECASE,
)
PYTHON_BLOCKED_PATTERN = re.compile(
    r"\b(import\s+os|from\s+os|subprocess|shutil|socket|requests|urllib|pathlib|yt[-_]?dlp|youtube[-_]?dl|pytube|streamlink|ffmpeg|pip\s+install|curl|wget|open\s*\(\s*['\"][/A-Za-z]:|open\s*\(\s*['\"].*\.\.)\b",
    re.IGNORECASE,
)


class AutomationService:
    def __init__(self, settings: Settings, llm: LlmService):
        self.settings = settings
        self.llm = llm

        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.automation_dir = base / "automation"
        self.runs_dir = self.automation_dir / "runs"
        self.browser_profile_dir = self.automation_dir / "browser-profile"
        self.downloads_dir = self.automation_dir / "downloads"
        self.workspace_dir = self.automation_dir / "workspace"
        self.recipes_path = self.automation_dir / "recipes.json"
        for path in [self.runs_dir, self.browser_profile_dir, self.downloads_dir, self.workspace_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.runs: dict[str, AutomationRun] = {}
        self._continue_events: dict[str, asyncio.Event] = {}
        self._confirmation_futures: dict[str, asyncio.Future[bool]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_download_processes: dict[str, subprocess.Popen[str]] = {}
        self._download_process_lock = threading.Lock()
        self._browser_lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser_context: Any = None
        self._page: Any = None

    async def start_run(self, request: AutomationRunRequest) -> AutomationRun:
        recipe = self.get_recipe(request.recipe_id) if request.recipe_id else None
        prompt = recipe.prompt if recipe and not request.prompt.strip() else request.prompt.strip()
        run = AutomationRun(id=uuid.uuid4().hex, prompt=prompt, recipe_id=recipe.id if recipe else request.recipe_id, create_recipe=request.create_recipe)
        self.runs[run.id] = run
        self._append_event(run, "queued", "Automation run queued.", {"prompt": run.prompt})
        self._save_run(run)
        asyncio.create_task(self._execute_run(run.id, recipe))
        return run

    def get_run(self, run_id: str) -> AutomationRun | None:
        if run_id in self.runs:
            return self.runs[run_id]
        path = self._run_path(run_id)
        if not path.exists():
            return None
        try:
            run = AutomationRun.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        self.runs[run.id] = run
        return run

    async def stream_events(self, run_id: str):
        cursor = 0
        while True:
            run = self.get_run(run_id)
            if not run:
                event = AutomationEvent(id=uuid.uuid4().hex, type="error", message="Automation run not found.")
                yield f"data: {event.model_dump_json()}\n\n"
                return
            while cursor < len(run.events):
                yield f"data: {run.events[cursor].model_dump_json()}\n\n"
                cursor += 1
            if run.status in {"complete", "error", "cancelled"}:
                return
            await asyncio.sleep(0.35)

    async def continue_run(self, run_id: str, request: AutomationContinueRequest) -> AutomationRun | None:
        run = self.get_run(run_id)
        if not run:
            return None
        event = self._continue_events.get(run_id)
        if event:
            event.set()
        self._set_status(run, "running")
        self._append_event(run, "continue", request.note or "User confirmed they are ready to continue.")
        self._save_run(run)
        return run

    async def cancel_run(self, run_id: str, request: AutomationCancelRequest) -> AutomationRun | None:
        run = self.get_run(run_id)
        if not run:
            return None
        if run.status in {"complete", "error", "cancelled"}:
            return run

        self._cancel_event_for_run(run_id).set()
        if continue_event := self._continue_events.get(run_id):
            continue_event.set()
        if confirmation_future := self._confirmation_futures.get(run_id):
            if not confirmation_future.done():
                confirmation_future.set_result(False)

        killed_process = self._kill_active_download_process(run_id)
        folder_path = str(self._safe_run_download_dir(run_id))
        self._cleanup_download_attempt_files(Path(folder_path), set())
        run.confirmation = None
        run.result = request.note or "Automation cancelled."
        self._set_status(run, "cancelled")
        if killed_process or any(event.type.startswith("download_") for event in run.events):
            self._append_event(
                run,
                "download_cancelled",
                "Download cancelled by user.",
                {"source_url": run.current_url, "folder_path": folder_path, "cancelled_by": "user"},
            )
        self._append_event(run, "cancelled", request.note or "Automation cancelled by user.", {"cancelled_by": "user"})
        self._save_run(run)
        return run

    async def confirm_run(self, run_id: str, request: AutomationConfirmRequest) -> AutomationRun | None:
        run = self.get_run(run_id)
        if not run:
            return None
        future = self._confirmation_futures.get(run_id)
        kind = run.confirmation.get("kind") if isinstance(run.confirmation, dict) else None
        accepted = request.approved
        message = "Approved." if request.approved else "Cancelled by user."
        if kind == "video_download_permission" and request.approved:
            accepted = request.confirmed_rights
            message = "Approved with rights confirmation." if accepted else "Video download permission was not confirmed."
        if future and not future.done():
            future.set_result(accepted)
        self._append_event(
            run,
            "confirmation",
            message,
            {"approved": request.approved, "accepted": accepted, "confirmed_rights": request.confirmed_rights, "attestation": request.attestation},
        )
        self._save_run(run)
        return run

    def list_recipes(self) -> list[AutomationRecipe]:
        return sorted(self._read_recipes(), key=lambda item: item.updated_at, reverse=True)

    def get_recipe(self, recipe_id: str | None) -> AutomationRecipe | None:
        if not recipe_id:
            return None
        return next((recipe for recipe in self._read_recipes() if recipe.id == recipe_id), None)

    def create_recipe(self, request: AutomationRecipeCreateRequest) -> AutomationRecipe:
        recipes = [recipe for recipe in self._read_recipes() if recipe.name.strip().lower() != request.name.strip().lower()]
        recipe = AutomationRecipe(id=uuid.uuid4().hex, name=request.name.strip(), prompt=request.prompt.strip(), steps=self._sanitize_steps(request.steps))
        recipes.append(recipe)
        self._write_recipes(recipes)
        return recipe

    def delete_recipe(self, recipe_id: str) -> bool:
        recipes = self._read_recipes()
        filtered = [recipe for recipe in recipes if recipe.id != recipe_id]
        if len(filtered) == len(recipes):
            return False
        self._write_recipes(filtered)
        return True

    def open_download_path(self, path: str) -> bool:
        try:
            target = Path(path).expanduser().resolve()
            downloads_root = self.downloads_dir.resolve()
            if target.is_file():
                target = target.parent
            if downloads_root not in target.parents and target != downloads_root:
                return False
            if not target.exists() or not target.is_dir():
                return False
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
            return True
        except Exception:
            return False

    async def _execute_run(self, run_id: str, recipe: AutomationRecipe | None = None) -> None:
        run = self.get_run(run_id)
        if not run:
            return
        plan: list[dict[str, Any]] = []
        try:
            self._set_status(run, "planning")
            self._append_event(run, "planning", "Planning automation steps.")
            if BLOCKED_PROMPT_PATTERN.search(run.prompt):
                raise ValueError("This automation request is blocked by the safety policy.")

            plan = recipe.steps if recipe and recipe.steps else await self._plan_steps(run.prompt)
            plan = self._sanitize_steps(plan)
            if not plan:
                raise ValueError("Astra could not create runnable automation steps.")

            self._set_status(run, "running")
            self._append_event(run, "plan_ready", f"Prepared {len(plan)} automation steps.", {"steps": plan})
            for step in plan:
                await self._execute_step(run, step)
                if run.status in {"cancelled", "error"}:
                    return

            if run.create_recipe or any(step["tool"] == "automation.save_recipe" for step in plan):
                recipe = self.create_recipe(
                    AutomationRecipeCreateRequest(name=self._recipe_name(run.prompt), prompt=run.prompt, steps=[step for step in plan if step["tool"] != "automation.save_recipe"])
                )
                run.recipe_id = recipe.id
                self._append_event(run, "recipe_saved", f"Saved automation: {recipe.name}.", {"recipe": recipe.model_dump(mode="json")})

            run.result = run.result or "Automation completed."
            self._set_status(run, "complete")
            self._append_event(run, "complete", run.result, {"current_url": run.current_url})
        except AutomationCancelledError:
            if run.status != "cancelled":
                run.result = run.result or "Automation cancelled."
                self._set_status(run, "cancelled")
                self._append_event(run, "cancelled", run.result)
        except Exception as exc:
            run.error = self._format_exception(exc)
            self._set_status(run, "error")
            self._append_event(run, "error", run.error)
        finally:
            self._save_run(run)
            if run.status in {"complete", "error", "cancelled"}:
                self._cancel_events.pop(run.id, None)
                self._continue_events.pop(run.id, None)
                self._confirmation_futures.pop(run.id, None)
                self._clear_active_download_process(run.id)

    async def _plan_steps(self, prompt: str) -> list[dict[str, Any]]:
        if self._should_use_stable_plan(prompt):
            return self._heuristic_plan(prompt)
        if self.settings.has_cerebras:
            try:
                llm_steps = await self._plan_with_llm(prompt)
                if llm_steps:
                    return llm_steps
            except Exception:
                pass
        return self._heuristic_plan(prompt)

    def _should_use_stable_plan(self, prompt: str) -> bool:
        normalized = prompt.lower()
        if "youtube" in normalized or "you tube" in normalized or "yt " in normalized:
            return True
        if "gmail" in normalized or "mail.google" in normalized:
            return True
        return bool(self._extract_url(prompt))

    async def _plan_with_llm(self, prompt: str) -> list[dict[str, Any]]:
        system_prompt = (
            "You are Astra's automation planner. Return only strict JSON with a top-level steps array. "
            f"Allowed tool values: {sorted(ALLOWED_AUTOMATION_TOOLS)}. "
            "Each step must be {tool, description, args}. Keep actions safe and ask for user/login waits when needed. "
            "Never output tools outside the enum."
        )
        user_prompt = f"User automation request: {prompt}"
        raw, _setup = await self.llm.complete(system_prompt, user_prompt, model=self.settings.resolved_cerebras_pro_model)
        payload = self._extract_json(raw)
        steps = payload.get("steps") if isinstance(payload, dict) else None
        return steps if isinstance(steps, list) else []

    def _heuristic_plan(self, prompt: str) -> list[dict[str, Any]]:
        normalized = prompt.lower()
        steps: list[dict[str, Any]] = []
        create_recipe = "new automation" in normalized or "reusable automation" in normalized or "save" in normalized

        if url := self._extract_url(prompt):
            steps.append({"tool": "browser.open", "description": f"Open {url}.", "args": {"url": url}})
            if "download" in normalized:
                if self._looks_like_direct_file(url) and not self._is_youtube_url(url):
                    steps.append({"tool": "browser.download", "description": "Download from the opened direct URL.", "args": {"url": url}})
                else:
                    steps.append({"tool": "video.download_permitted", "description": "Download the opened video after rights confirmation.", "args": {"url": url}})
            else:
                steps.append({"tool": "browser.extract", "description": "Extract page content.", "args": {"target": "page content"}})
        elif "gmail" in normalized or "mail.google" in normalized:
            steps.append({"tool": "browser.open", "description": "Open Gmail.", "args": {"url": "https://mail.google.com/"}})
            steps.append({"tool": "browser.extract", "description": "Read visible Gmail information.", "args": {"target": "first 5 emails"}})
        elif "youtube" in normalized or "you tube" in normalized or "yt " in normalized:
            query = self._extract_search_query(prompt, ("youtube", "you tube", "yt", "open", "download", "play", "search", "video"))
            if query:
                steps.append({"tool": "browser.search", "description": "Search YouTube.", "args": {"site": "youtube", "query": query}})
            else:
                steps.append({"tool": "browser.open", "description": "Open YouTube.", "args": {"url": "https://www.youtube.com/"}})
            if "download" in normalized:
                steps.append({"tool": "browser.click", "description": "Open the first visible YouTube result.", "args": {"selector": "ytd-video-renderer a#thumbnail"}})
                steps.append({"tool": "video.download_permitted", "description": "Download the opened video after rights confirmation.", "args": {}})
            elif "play" in normalized:
                steps.append({"tool": "browser.click", "description": "Open the first visible result.", "args": {"selector": "ytd-video-renderer a#thumbnail"}})
            else:
                steps.append({"tool": "browser.extract", "description": "Extract visible video results.", "args": {"target": "search results"}})
        elif "python" in normalized:
            steps.append({"tool": "python.run_safe", "description": "Run a constrained Python task.", "args": {"code": "print('Describe the Python automation steps more specifically.')"}})
        else:
            query = self._extract_search_query(prompt, ("open", "search", "find", "tell", "me", "about"))
            steps.append({"tool": "browser.search", "description": "Search the web.", "args": {"site": "google", "query": query or prompt}})
            steps.append({"tool": "browser.extract", "description": "Extract visible result information.", "args": {"target": "search results"}})

        if create_recipe:
            steps.append({"tool": "automation.save_recipe", "description": "Save this workflow for reuse.", "args": {}})
        return steps

    def _sanitize_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean_steps: list[dict[str, Any]] = []
        for raw in steps:
            if not isinstance(raw, dict):
                continue
            tool = str(raw.get("tool") or "").strip()
            if tool not in ALLOWED_AUTOMATION_TOOLS:
                continue
            args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
            description = str(raw.get("description") or tool).strip()[:240]
            if tool == "browser.open":
                url = str(args.get("url") or "").strip()
                if not self._valid_http_url(url):
                    continue
                args["url"] = url
            if tool == "browser.search":
                args["query"] = str(args.get("query") or "").strip()[:240]
                args["site"] = str(args.get("site") or "google").strip().lower()[:40]
                if not args["query"]:
                    continue
            if tool == "python.run_safe":
                args["code"] = str(args.get("code") or "").strip()
                if not args["code"] or PYTHON_BLOCKED_PATTERN.search(args["code"]):
                    continue
            if tool == "video.download_permitted":
                url = str(args.get("url") or "").strip()
                if url and not self._valid_http_url(url):
                    continue
                if url:
                    args["url"] = url
            clean_steps.append({"tool": tool, "description": description, "args": args})
        return clean_steps

    async def _execute_step(self, run: AutomationRun, step: dict[str, Any]) -> None:
        tool = step["tool"]
        if tool == "automation.save_recipe":
            self._append_event(run, "step", "Recipe will be saved after successful completion.", {"step": step})
            return

        if self._requires_confirmation(run, step):
            approved = await self._wait_for_confirmation(run, step)
            if not approved:
                run.result = "Automation cancelled."
                self._set_status(run, "cancelled")
                self._append_event(run, "cancelled", "Automation cancelled before a sensitive action.")
                return

        self._append_event(run, "step", step["description"], {"step": step})
        if tool == "browser.open":
            await self._tool_browser_open(run, step["args"]["url"])
        elif tool == "browser.search":
            await self._tool_browser_search(run, step["args"])
        elif tool == "browser.click":
            await self._tool_browser_click(run, step["args"])
        elif tool == "browser.type":
            await self._tool_browser_type(run, step["args"])
        elif tool == "browser.extract":
            await self._tool_browser_extract(run, step["args"])
        elif tool == "browser.wait_for_user":
            await self._wait_for_user(run, str(step["args"].get("reason") or step["description"]))
        elif tool == "browser.download":
            await self._tool_browser_download(run, step["args"])
        elif tool == "python.run_safe":
            await self._tool_python_run_safe(run, step["args"]["code"])
        elif tool == "video.download_permitted":
            await self._tool_video_download_permitted(run, step["args"])

    async def _tool_browser_open(self, run: AutomationRun, url: str) -> None:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                page = await self._ensure_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                run.current_url = page.url
                self._append_event(run, "browser", f"Opened {page.url}.", {"url": page.url})
                await self._wait_for_login_if_needed(run)
                return
            except Exception as exc:
                last_error = exc
                await self._reset_browser_handles()
                if attempt == 0:
                    self._append_event(run, "browser_recovery", "Restarting the Automation Browser and retrying the page open.")
                    continue
                break
        raise RuntimeError(f"Automation Browser could not open {url}: {self._format_exception(last_error)}") from last_error

    async def _tool_browser_search(self, run: AutomationRun, args: dict[str, Any]) -> None:
        site = str(args.get("site") or "google").lower()
        query = str(args.get("query") or "").strip()
        if site == "youtube":
            url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        else:
            url = f"https://www.google.com/search?q={quote_plus(query)}"
        await self._tool_browser_open(run, url)

    async def _tool_browser_click(self, run: AutomationRun, args: dict[str, Any]) -> None:
        page = await self._ensure_page()
        selector = str(args.get("selector") or "").strip()
        text = str(args.get("text") or "").strip()
        if selector:
            try:
                await page.click(selector, timeout=10000)
            except Exception:
                await self._click_with_selector_fallbacks(page, selector)
        elif text:
            await page.get_by_text(text, exact=False).first.click(timeout=10000)
        else:
            raise ValueError("Click step needs a selector or text.")
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        run.current_url = page.url
        self._append_event(run, "browser", "Clicked requested page element.", {"url": page.url})
        await self._wait_for_login_if_needed(run)

    async def _click_with_selector_fallbacks(self, page: Any, selector: str) -> None:
        selectors: list[str] = []
        if "ytd-video-renderer" in selector or "youtube" in page.url.lower():
            selectors.extend(["a#video-title", "ytd-video-renderer a[href*='/watch']", "a[href*='/watch']"])

        last_error: Exception | None = None
        for fallback_selector in selectors:
            try:
                await page.locator(fallback_selector).first.click(timeout=10000)
                return
            except Exception as exc:
                last_error = exc
        raise ValueError(f"Could not click the requested page element: {selector}") from last_error

    async def _tool_browser_type(self, run: AutomationRun, args: dict[str, Any]) -> None:
        page = await self._ensure_page()
        selector = str(args.get("selector") or "").strip()
        text = str(args.get("text") or "").strip()
        if not selector or not text:
            raise ValueError("Type step needs selector and text.")
        await page.fill(selector, text, timeout=10000)
        run.current_url = page.url
        self._append_event(run, "browser", "Typed into page field.", {"url": page.url})

    async def _tool_browser_extract(self, run: AutomationRun, args: dict[str, Any]) -> None:
        page = await self._ensure_page()
        await self._wait_for_login_if_needed(run)
        title = await page.title()
        body = await page.locator("body").inner_text(timeout=10000)
        target = str(args.get("target") or "page content")
        excerpt = self._compact_text(body)[:3000]
        run.current_url = page.url
        run.result = f"{target}: {title}\n\n{excerpt}".strip()
        self._append_event(run, "extract", f"Extracted {target}.", {"title": title, "url": page.url, "text": excerpt})

    async def _tool_browser_download(self, run: AutomationRun, args: dict[str, Any]) -> None:
        page = await self._ensure_page()
        url = str(args.get("url") or "").strip()
        if url and self._valid_http_url(url) and self._looks_like_direct_file(url):
            response = await page.context.request.get(url, timeout=30000)
            if not response.ok:
                raise ValueError(f"Download failed with HTTP {response.status}.")
            filename = self._safe_filename(Path(urlparse(url).path).name or f"automation-{uuid.uuid4().hex[:8]}")
            target = self.downloads_dir / filename
            target.write_bytes(await response.body())
            run.result = f"Downloaded {target.name}."
            self._append_event(run, "download", run.result, {"path": str(target), "filename": target.name})
            return
        else:
            try:
                async with page.expect_download(timeout=30000) as download_info:
                    await self._click_official_download_control(page, str(args.get("selector") or "").strip())
                download = await download_info.value
            except Exception:
                run.result = self._download_unavailable_message(page.url)
                self._append_event(run, "download_unavailable", run.result, {"url": page.url, "copyable_url": page.url})
                return
        suggested = download.suggested_filename or f"automation-{uuid.uuid4().hex[:8]}"
        target = self.downloads_dir / self._safe_filename(suggested)
        await download.save_as(str(target))
        run.result = f"Downloaded {target.name}."
        self._append_event(run, "download", run.result, {"path": str(target), "filename": target.name})

    async def _click_official_download_control(self, page: Any, selector: str) -> None:
        if selector:
            await page.click(selector, timeout=10000)
            return

        locators = [
            page.get_by_role("button", name=re.compile("download", re.IGNORECASE)).first,
            page.get_by_role("link", name=re.compile("download", re.IGNORECASE)).first,
            page.get_by_text("Download", exact=False).first,
        ]
        last_error: Exception | None = None
        for locator in locators:
            try:
                await locator.click(timeout=5000)
                return
            except Exception as exc:
                last_error = exc
        raise ValueError("No official browser download control was found.") from last_error

    def _download_unavailable_message(self, page_url: str) -> str:
        host = urlparse(page_url).netloc.lower()
        if host.endswith("youtube.com") or host.endswith("youtu.be"):
            return (
                "No official browser download control was available on this YouTube page. "
                f"Video link: {page_url}"
            )
        return f"No official browser download control was available on this page. Link: {page_url}"

    async def _tool_video_download_permitted(self, run: AutomationRun, args: dict[str, Any]) -> None:
        source_url = str(args.get("url") or run.current_url).strip()
        if not source_url:
            page = await self._ensure_page()
            source_url = page.url
        if not self._valid_http_url(source_url):
            raise ValueError("Video download needs a valid video URL.")
        if self._url_requires_login(source_url):
            raise ValueError("Astra will not download videos from login-required pages.")

        try:
            self._raise_if_cancelled(run.id)
            result = await asyncio.wait_for(
                asyncio.to_thread(self._download_video_with_ytdlp, run, source_url, VIDEO_DOWNLOAD_MAX_SIZE_BYTES),
                timeout=VIDEO_DOWNLOAD_TIMEOUT_SECONDS + 30,
            )
        except asyncio.TimeoutError as exc:
            folder_path = str(self._safe_run_download_dir(run.id))
            message = "Video download timed out before completion."
            self._append_event(run, "download_error", message, {"source_url": source_url, "folder_path": folder_path})
            raise ValueError(message) from exc
        except AutomationCancelledError:
            folder_path = str(self._safe_run_download_dir(run.id))
            if run.status != "cancelled":
                run.result = "Automation cancelled."
                self._set_status(run, "cancelled")
                self._append_event(run, "download_cancelled", "Download cancelled by user.", {"source_url": source_url, "folder_path": folder_path, "cancelled_by": "user"})
            raise
        except Exception as exc:
            folder_path = str(self._safe_run_download_dir(run.id))
            message = f"Video download failed: {self._friendly_download_error(exc)}"
            self._append_event(run, "download_error", message, {"source_url": source_url, "folder_path": folder_path})
            raise ValueError(message) from exc

        run.result = f"Downloaded {result['filename']}."
        self._append_event(run, "download_complete", run.result, result)

    def _download_video_with_ytdlp(self, run: AutomationRun, source_url: str, max_size_bytes: int) -> dict[str, Any]:
        self._raise_if_cancelled(run.id)
        try:
            import yt_dlp
        except Exception as exc:
            raise RuntimeError("yt-dlp is not installed. Install backend requirements, then restart Astra.") from exc

        run_download_dir = self._safe_run_download_dir(run.id)
        folder_path = str(run_download_dir)
        before_files = {path.resolve() for path in run_download_dir.glob("*") if path.is_file()}
        last_progress_percent = {"value": -1}

        def progress_hook(progress: dict[str, Any]) -> None:
            self._raise_if_cancelled(run.id)
            if progress.get("status") != "downloading":
                return
            downloaded = int(progress.get("downloaded_bytes") or 0)
            total = int(progress.get("total_bytes") or progress.get("total_bytes_estimate") or 0)
            percent = int(progress.get("progress_percent") or ((downloaded / total) * 100 if total else 0))
            percent = max(0, min(percent, 100))
            if percent == last_progress_percent["value"]:
                return
            last_progress_percent["value"] = percent
            self._append_event(
                run,
                "download_progress",
                f"Downloading video: {percent}%.",
                {
                    "source_url": source_url,
                    "folder_path": folder_path,
                    "progress_percent": percent,
                    "downloaded_bytes": downloaded or None,
                    "total_bytes": total or None,
                },
            )

        probe_options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "format": VIDEO_DOWNLOAD_FORMAT_SELECTOR,
            "merge_output_format": "mkv",
        }
        if ffmpeg_location := self._ffmpeg_location():
            probe_options["ffmpeg_location"] = ffmpeg_location
        with yt_dlp.YoutubeDL(probe_options) as ydl:
            info = ydl.extract_info(source_url, download=False)

        self._raise_if_cancelled(run.id)
        info = self._normalize_video_info(info)
        self._validate_video_info(info, source_url, max_size_bytes)
        title = str(info.get("title") or "video").strip() or "video"
        self._append_event(
            run,
            "download_start",
            f"Downloading {title}.",
            {
                "source_url": source_url,
                "folder_path": folder_path,
                "progress_percent": 0,
                "downloaded_bytes": 0,
                "total_bytes": self._selected_video_size(info) or None,
                "video_title": title,
                "channel": info.get("channel") or info.get("uploader") or "",
                "max_size_mb": VIDEO_DOWNLOAD_MAX_SIZE_MB,
            },
        )

        try:
            self._run_ytdlp_download_subprocess(run, source_url, run_download_dir, before_files, max_size_bytes, progress_hook)
        except AutomationCancelledError:
            self._cleanup_download_attempt_files(run_download_dir, before_files)
            raise

        self._raise_if_cancelled(run.id)
        target = self._latest_download_file(run_download_dir, before_files)
        return {
            "source_url": source_url,
            "folder_path": folder_path,
            "progress_percent": 100,
            "downloaded_bytes": target.stat().st_size,
            "total_bytes": target.stat().st_size,
            "filename": target.name,
            "path": str(target),
            "video_title": title,
            "channel": info.get("channel") or info.get("uploader") or "",
        }

    def _run_ytdlp_download_subprocess(
        self,
        run: AutomationRun,
        source_url: str,
        run_download_dir: Path,
        before_files: set[Path],
        max_size_bytes: int,
        progress_hook: Any,
    ) -> None:
        attempts = [
            {
                "label": "best available quality",
                "format_selector": VIDEO_DOWNLOAD_FORMAT_SELECTOR,
                "format_sort": "proto:https,res,fps",
                "merge_output_format": "mkv",
            },
            {
                "label": "compatible MP4 fallback",
                "format_selector": VIDEO_DOWNLOAD_FALLBACK_FORMAT_SELECTOR,
                "format_sort": "proto:https,res,fps",
                "merge_output_format": "",
            },
        ]
        failures: list[str] = []
        try:
            for index, attempt in enumerate(attempts):
                self._raise_if_cancelled(run.id)
                if index > 0:
                    self._cleanup_download_attempt_files(run_download_dir, before_files)
                    self._append_event(
                        run,
                        "download_retry",
                        f"Top-quality stream failed; retrying with {attempt['label']}.",
                        {"source_url": source_url, "folder_path": str(run_download_dir), "attempt": attempt["label"], "previous_error": failures[-1] if failures else ""},
                    )
                command = self._ytdlp_download_command(
                    source_url,
                    run_download_dir,
                    max_size_bytes,
                    format_selector=str(attempt["format_selector"]),
                    format_sort=str(attempt["format_sort"]),
                    merge_output_format=str(attempt["merge_output_format"]),
                )
                try:
                    self._run_ytdlp_command_once(run.id, command, run_download_dir, progress_hook)
                    return
                except AutomationCancelledError:
                    raise
                except TimeoutError:
                    raise
                except Exception as exc:
                    failures.append(self._summarize_ytdlp_failure(str(exc)) or self._format_exception(exc))
                    if index == len(attempts) - 1:
                        raise ValueError("\n".join(failures[-2:])) from exc
        except AutomationCancelledError:
            self._cleanup_download_attempt_files(run_download_dir, before_files)
            raise
        raise ValueError("\n".join(failures) or "yt-dlp download failed.")

    def _run_ytdlp_command_once(self, run_id: str, command: list[str], run_download_dir: Path, progress_hook: Any) -> None:
        self._raise_if_cancelled(run_id)
        process = subprocess.Popen(
            command,
            cwd=run_download_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=False,
        )
        self._set_active_download_process(run_id, process)
        started_at = time.monotonic()
        output_lines: list[str] = []
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if self._is_cancelled(run_id):
                    process.kill()
                    raise AutomationCancelledError("Download cancelled by user.")
                output_lines.append(line)
                progress = self._parse_ytdlp_progress_line(line)
                if progress:
                    progress_hook(progress)
                if time.monotonic() - started_at > VIDEO_DOWNLOAD_TIMEOUT_SECONDS:
                    process.kill()
                    raise TimeoutError("yt-dlp download timed out.")
            returncode = process.wait(timeout=5)
            if self._is_cancelled(run_id):
                raise AutomationCancelledError("Download cancelled by user.")
        finally:
            if process.poll() is None:
                process.kill()
            self._clear_active_download_process(run_id, process)
        if returncode != 0:
            output = "".join(output_lines).strip()
            raise ValueError(self._summarize_ytdlp_failure(output) or f"yt-dlp exited with code {returncode}.")

    def _ytdlp_download_command(
        self,
        source_url: str,
        run_download_dir: Path,
        max_size_bytes: int,
        format_selector: str = VIDEO_DOWNLOAD_FORMAT_SELECTOR,
        format_sort: str = "proto:https,res,fps",
        merge_output_format: str = "mkv",
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-update",
            "--no-playlist",
            "--newline",
            "--progress",
            "--progress-delta",
            "0.25",
            "--progress-template",
            "download:ASTRA_PROGRESS:%(progress._percent_str)s|%(progress.downloaded_bytes)s|%(progress.total_bytes)s|%(progress.total_bytes_estimate)s",
            "--max-filesize",
            str(max_size_bytes),
            "-f",
            format_selector,
            "--format-sort",
            format_sort,
            "-o",
            str(run_download_dir / "%(title).180B [%(id)s].%(ext)s"),
            source_url,
        ]
        if merge_output_format:
            command[-3:-3] = ["--merge-output-format", merge_output_format]
        if merge_output_format and (ffmpeg_location := self._ffmpeg_location()):
            command[3:3] = ["--ffmpeg-location", ffmpeg_location]
        return command

    def _cleanup_download_attempt_files(self, run_download_dir: Path, before_files: set[Path]) -> None:
        for path in run_download_dir.glob("*"):
            try:
                resolved = path.resolve()
                if not path.is_file() or resolved in before_files or run_download_dir.resolve() not in resolved.parents:
                    continue
                path.unlink(missing_ok=True)
            except Exception:
                pass

    def _cancel_event_for_run(self, run_id: str) -> threading.Event:
        event = self._cancel_events.get(run_id)
        if event is None:
            event = threading.Event()
            self._cancel_events[run_id] = event
        return event

    def _is_cancelled(self, run_id: str) -> bool:
        event = self._cancel_events.get(run_id)
        return bool(event and event.is_set())

    def _raise_if_cancelled(self, run_id: str) -> None:
        if self._is_cancelled(run_id):
            raise AutomationCancelledError("Download cancelled by user.")

    def _set_active_download_process(self, run_id: str, process: subprocess.Popen[str]) -> None:
        with self._download_process_lock:
            self._active_download_processes[run_id] = process

    def _clear_active_download_process(self, run_id: str, process: subprocess.Popen[str] | None = None) -> None:
        with self._download_process_lock:
            current = self._active_download_processes.get(run_id)
            if process is None or current is process:
                self._active_download_processes.pop(run_id, None)

    def _kill_active_download_process(self, run_id: str) -> bool:
        with self._download_process_lock:
            process = self._active_download_processes.get(run_id)
        if not process or process.poll() is not None:
            return False
        try:
            process.kill()
            return True
        except Exception:
            return False

    def _ffmpeg_location(self) -> str | None:
        try:
            import imageio_ffmpeg

            return str(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            return None

    def _summarize_ytdlp_failure(self, output: str) -> str:
        clean = re.sub(r"\x1b\[[0-9;]*m", "", output).strip()
        if not clean:
            return ""
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        important = [
            line
            for line in lines
            if "ERROR:" in line
            or "HTTP Error 403" in line
            or "Requested format is not available" in line
            or "file is empty" in line
            or "max-filesize" in line.lower()
            or "Postprocessing" in line
            or "Conversion failed" in line
        ]
        selected = important[-8:] if important else lines[-8:]
        return "\n".join(selected)[-1000:]

    def _friendly_download_error(self, error: Exception) -> str:
        message = self._summarize_ytdlp_failure(str(error)) or self._compact_text(str(error))[-1000:]
        if "HTTP Error 403" in message or "Forbidden" in message:
            return (
                "The video host denied the selected stream with HTTP 403. "
                "Astra tried best available quality and the compatible MP4 fallback. "
                f"Last downloader details: {message}"
            )
        if "Requested format is not available" in message:
            return (
                "No compatible video format was available for this video. "
                f"Last downloader details: {message}"
            )
        return message

    def _parse_ytdlp_progress_line(self, line: str) -> dict[str, Any] | None:
        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
        structured = re.search(r"ASTRA_PROGRESS:([^|]+)\|([^|]*)\|([^|]*)\|([^|]*)", clean_line)
        if structured:
            percent_match = re.search(r"(\d+(?:\.\d+)?)", structured.group(1))
            if not percent_match:
                return None
            total_bytes = self._parse_optional_int(structured.group(3)) or self._parse_optional_int(structured.group(4))
            return {
                "status": "downloading",
                "progress_percent": int(float(percent_match.group(1))),
                "downloaded_bytes": self._parse_optional_int(structured.group(2)),
                "total_bytes": total_bytes,
            }
        percent_match = re.search(r"(\d+(?:\.\d+)?)%", clean_line)
        if not percent_match:
            return None
        return {"status": "downloading", "progress_percent": int(float(percent_match.group(1)))}

    def _parse_optional_int(self, value: str) -> int | None:
        cleaned = value.strip()
        if not cleaned or cleaned.upper() in {"NA", "N/A", "NONE", "NULL"}:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None

    def _normalize_video_info(self, info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            raise ValueError("The video metadata could not be read.")
        if info.get("_type") == "playlist" or info.get("entries"):
            raise ValueError("Playlists are not supported in this automation. Ask for one specific video.")
        return info

    def _validate_video_info(self, info: dict[str, Any], source_url: str, max_size_bytes: int) -> None:
        availability = str(info.get("availability") or "").strip().lower()
        if availability in BLOCKED_VIDEO_AVAILABILITY:
            raise ValueError(f"This video cannot be downloaded because its availability is {availability}.")
        if self._is_youtube_url(source_url) and availability != "public":
            raise ValueError("Astra only downloads YouTube videos that yt-dlp verifies as public.")

        filesize = self._selected_video_size(info)
        if isinstance(filesize, (int, float)) and filesize > max_size_bytes:
            raise ValueError(f"This video is larger than the {VIDEO_DOWNLOAD_MAX_SIZE_MB} MB automation limit.")

    def _selected_video_size(self, info: dict[str, Any]) -> int | None:
        filesize = info.get("filesize") or info.get("filesize_approx")
        if isinstance(filesize, (int, float)) and filesize > 0:
            return int(filesize)

        requested_downloads = info.get("requested_downloads")
        if not isinstance(requested_downloads, list):
            return None
        total = 0
        for item in requested_downloads:
            if not isinstance(item, dict):
                continue
            item_size = item.get("filesize") or item.get("filesize_approx")
            if not isinstance(item_size, (int, float)) or item_size <= 0:
                return None
            total += int(item_size)
        return total or None

    def _safe_run_download_dir(self, run_id: str) -> Path:
        safe_run_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_id)[:80] or uuid.uuid4().hex
        run_download_dir = (self.downloads_dir / safe_run_id).resolve()
        downloads_root = self.downloads_dir.resolve()
        if downloads_root not in run_download_dir.parents and run_download_dir != downloads_root:
            raise ValueError("Download path escaped the automation downloads folder.")
        run_download_dir.mkdir(parents=True, exist_ok=True)
        return run_download_dir

    def _latest_download_file(self, run_download_dir: Path, before_files: set[Path]) -> Path:
        candidates = [
            path.resolve()
            for path in run_download_dir.glob("*")
            if path.is_file() and not path.name.endswith((".part", ".ytdl")) and path.resolve() not in before_files
        ]
        if not candidates:
            raise ValueError("yt-dlp finished without producing a downloaded file.")
        target = max(candidates, key=lambda path: path.stat().st_mtime)
        if run_download_dir.resolve() not in target.parents:
            raise ValueError("Downloaded file path escaped the run download folder.")
        return target

    async def _tool_python_run_safe(self, run: AutomationRun, code: str) -> None:
        if PYTHON_BLOCKED_PATTERN.search(code):
            raise ValueError("Python code was blocked by the automation sandbox.")

        script_path = self.workspace_dir / f"run-{run.id}.py"
        script_path.write_text(code, encoding="utf-8")

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, "ASTRA_AUTOMATION_DOWNLOADS": str(self.downloads_dir)},
            )

        completed = await asyncio.to_thread(execute)
        output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        if completed.returncode != 0:
            raise ValueError(f"Python automation failed with exit code {completed.returncode}: {output[-1200:]}")
        run.result = output[-3000:] or "Python automation completed."
        self._append_event(run, "python", "Python automation completed.", {"output": output[-3000:]})

    async def _ensure_page(self):
        async with self._browser_lock:
            if self._page and not self._page.is_closed():
                return self._page
            try:
                from playwright.async_api import async_playwright
            except Exception as exc:
                raise RuntimeError(
                    "Playwright is not installed. Install it with `pip install playwright` and `python -m playwright install chromium`."
                ) from exc
            if not self._playwright:
                self._playwright = await async_playwright().start()
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    if not self._browser_context:
                        self._browser_context = await self._playwright.chromium.launch_persistent_context(
                            user_data_dir=str(self.browser_profile_dir),
                            headless=False,
                            accept_downloads=True,
                            downloads_path=str(self.downloads_dir),
                        )
                    pages = [page for page in self._browser_context.pages if not page.is_closed()]
                    self._page = pages[0] if pages else await self._browser_context.new_page()
                    return self._page
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        await self._reset_browser_handles()
                        continue
                    break
            raise RuntimeError(f"Could not open the Automation Browser: {self._format_exception(last_error)}") from last_error

    async def _reset_browser_handles(self) -> None:
        context = self._browser_context
        self._page = None
        self._browser_context = None
        if context:
            try:
                await context.close()
            except Exception:
                pass

    def _is_closed_browser_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "target page" in message and ("closed" in message or "browser has been closed" in message)

    async def _wait_for_login_if_needed(self, run: AutomationRun) -> None:
        page = await self._ensure_page()
        current = page.url
        host = urlparse(current).netloc.lower()
        if "accounts.google.com" in host or "signin" in current.lower() or "login" in current.lower():
            await self._wait_for_user(run, "Log in in the Automation Browser, then press Continue.")

    async def _wait_for_user(self, run: AutomationRun, reason: str) -> None:
        event = self._continue_events.get(run.id)
        if not event:
            event = asyncio.Event()
            self._continue_events[run.id] = event
        event.clear()
        self._set_status(run, "waiting_for_login")
        self._append_event(run, "waiting_for_login", reason)
        self._save_run(run)
        await event.wait()
        self._set_status(run, "running")
        self._append_event(run, "running", "Continuing automation.")

    async def _wait_for_confirmation(self, run: AutomationRun, step: dict[str, Any]) -> bool:
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._confirmation_futures[run.id] = future
        run.confirmation = await self._confirmation_payload(run, step)
        self._set_status(run, "confirmation_required")
        self._append_event(run, "confirmation_required", run.confirmation["message"], run.confirmation)
        self._save_run(run)
        approved = await future
        run.confirmation = None
        self._set_status(run, "running")
        self._save_run(run)
        return approved

    def _requires_confirmation(self, run: AutomationRun, step: dict[str, Any]) -> bool:
        tool = step["tool"]
        if tool in {"browser.download", "python.run_safe", "video.download_permitted"}:
            return True
        host = urlparse(run.current_url).netloc.lower()
        if any(host.endswith(private_host) for private_host in PRIVATE_HOSTS) and tool in {"browser.click", "browser.type", "browser.extract"}:
            return True
        return False

    async def _confirmation_payload(self, run: AutomationRun, step: dict[str, Any]) -> dict[str, Any]:
        if step["tool"] == "video.download_permitted":
            return await self._video_download_confirmation_payload(run, step)
        return {"step": step, "message": self._confirmation_message(step)}

    async def _video_download_confirmation_payload(self, run: AutomationRun, step: dict[str, Any]) -> dict[str, Any]:
        page = await self._ensure_page()
        video_url = str(step.get("args", {}).get("url") or run.current_url or page.url).strip()
        title = ""
        channel = ""
        try:
            title = (await page.title()).replace(" - YouTube", "").strip()
        except Exception:
            title = ""
        try:
            channel = (await page.locator("ytd-video-owner-renderer #channel-name a").first.inner_text(timeout=2000)).strip()
        except Exception:
            channel = ""
        rights_statement = "I own this video or have permission/license to download it."
        return {
            "kind": "video_download_permission",
            "step": step,
            "message": "Astra needs your rights confirmation before downloading this video.",
            "video_url": video_url,
            "video_title": title,
            "channel": channel,
            "rights_statement": rights_statement,
            "max_size_mb": VIDEO_DOWNLOAD_MAX_SIZE_MB,
        }

    def _confirmation_message(self, step: dict[str, Any]) -> str:
        if step["tool"] == "python.run_safe":
            return "Astra needs approval before running Python for this automation."
        if step["tool"] == "video.download_permitted":
            return "Astra needs your rights confirmation before downloading this video."
        if step["tool"] == "browser.download":
            return "Astra needs approval before downloading a file."
        return f"Astra needs approval before: {step['description']}"

    def _set_status(self, run: AutomationRun, status: str) -> None:
        run.status = status  # type: ignore[assignment]
        run.updated_at = datetime.utcnow()
        self._save_run(run)

    def _append_event(self, run: AutomationRun, event_type: str, message: str, data: dict[str, Any] | None = None) -> AutomationEvent:
        event = AutomationEvent(id=uuid.uuid4().hex, type=event_type, message=message, data=data or {})
        run.events.append(event)
        run.updated_at = datetime.utcnow()
        self._save_run(run)
        return event

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _save_run(self, run: AutomationRun) -> None:
        self.runs[run.id] = run
        self._run_path(run.id).write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def _read_recipes(self) -> list[AutomationRecipe]:
        if not self.recipes_path.exists():
            return []
        try:
            payload = json.loads(self.recipes_path.read_text(encoding="utf-8"))
            return [AutomationRecipe.model_validate(item) for item in payload if isinstance(item, dict)]
        except Exception:
            return []

    def _write_recipes(self, recipes: list[AutomationRecipe]) -> None:
        payload = [recipe.model_dump(mode="json") for recipe in recipes]
        self.recipes_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _extract_json(self, text: Any) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {}
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}

    def _extract_url(self, text: str) -> str | None:
        match = re.search(r"https?://[^\s]+", text, re.IGNORECASE)
        if not match:
            return None
        url = match.group(0).rstrip(".,)")
        return url if self._valid_http_url(url) else None

    def _valid_http_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _is_youtube_url(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return host.endswith("youtube.com") or host.endswith("youtu.be")

    def _url_requires_login(self, url: str) -> bool:
        parsed = urlparse(url)
        current = url.lower()
        host = parsed.netloc.lower()
        return "accounts.google.com" in host or "signin" in current or "login" in current

    def _looks_like_direct_file(self, url: str) -> bool:
        return bool(re.search(r"\.(pdf|zip|csv|json|txt|md|png|jpg|jpeg|webp|mp3|mp4)(\?|$)", url, re.IGNORECASE))

    def _extract_search_query(self, prompt: str, stop_words: tuple[str, ...]) -> str:
        cleaned = prompt.replace("it's", " ").replace("it’s", " ")
        for word in stop_words:
            cleaned = re.sub(rf"\b{re.escape(word)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(and|then|please|specific|video|videos|first|result|for|on|in|its)\b", " ", cleaned, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", cleaned).strip(" .")

    def _compact_text(self, text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()

    def _format_exception(self, exc: BaseException | None) -> str:
        if exc is None:
            return "Unknown automation error."
        message = self._compact_text(str(exc))
        if not message:
            message = repr(exc)
        if not message or message == f"{type(exc).__name__}()":
            message = type(exc).__name__
        return message[-1600:]

    def _safe_filename(self, filename: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9._ -]+", "_", filename).strip(" .")
        return cleaned or f"automation-{uuid.uuid4().hex[:8]}"

    def _recipe_name(self, prompt: str) -> str:
        words = re.sub(r"[^a-zA-Z0-9 ]+", " ", prompt).split()
        name = " ".join(words[:6]).strip() or "Automation"
        return name[:64]
