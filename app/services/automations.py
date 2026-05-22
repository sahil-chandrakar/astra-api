import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

from app.config import Settings
from app.models import (
    AutomationCancelRequest,
    AutomationConfirmRequest,
    AutomationContinueRequest,
    AutomationEngineStatus,
    AutomationEvent,
    AutomationArtifact,
    AutomationRecipe,
    AutomationRecipeCreateRequest,
    AutomationRun,
    AutomationRunRequest,
    AutomationSuggestion,
)
from app.services.llm import LlmService
from app.services.agent_runtime import AgentRuntime
from app.services.artifacts import ArtifactService
from app.services.automation_builder import AutomationBuilderService, FUTURE_DESKTOP_TOOLS
from app.services.openrpa import OpenRPAAdapter, OpenRPACancelledError, OpenRPAMissingError
from app.services.windows_automation import WindowsAutomationResult, WindowsAutomationService


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
    "youtube.search",
    "youtube.result",
    "artifact.resolve_reference",
    "artifact.list_recent",
    "artifact.pick",
    "artifact.open_containing_folder",
    "file.search_scoped",
    "file.open",
    "app.resolve",
    "app.open",
    "app.open_with_file",
    "desktop.find_text",
    "desktop.click",
    "desktop.type_text",
    "desktop.press_key",
    "desktop.verify_text",
    "python.run_safe",
    "automation.run_python_safe",
    "video.download",
    "video.download_permitted",
    "system.ask_user",
    "windows.set_alarm",
    "windows.open_calculator",
    "windows.open_file_explorer",
    "windows.prepare_whatsapp_message",
    "windows.send_prepared_whatsapp_message",
    "windows.reject_destructive_file_action",
    "automation.save_recipe",
    "task.finish",
    "task.replan",
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
OPENRPA_UNSAFE_METADATA_PATTERN = re.compile(
    r"\b(password|passcode|token|secret|credential|login|sign\s*in|credit\s*card|card\s*number|cvv|payment|pay|purchase|buy|checkout|"
    r"delete|remove|erase|format|wipe|empty\s+recycle\s+bin|destructive|steal|cookie|session)\b",
    re.IGNORECASE,
)
PYTHON_BLOCKED_PATTERN = re.compile(
    r"\b(import\s+os|from\s+os|subprocess|shutil|socket|requests|urllib|pathlib|yt[-_]?dlp|youtube[-_]?dl|pytube|streamlink|ffmpeg|pip\s+install|curl|wget|open\s*\(\s*['\"][/A-Za-z]:|open\s*\(\s*['\"].*\.\.)\b",
    re.IGNORECASE,
)
WINDOWS_DESTRUCTIVE_FILE_PATTERN = re.compile(
    r"\b(delete|remove|move|rename|format|wipe|empty\s+recycle\s+bin)\b.*\b(file|folder|directory|drive|explorer|path|desktop|downloads?|documents?)\b|"
    r"\b(file|folder|directory|drive|explorer|path|desktop|downloads?|documents?)\b.*\b(delete|remove|move|rename|format|wipe)\b",
    re.IGNORECASE,
)
CALCULATOR_EXPRESSION_PATTERN = re.compile(r"^[0-9+\-*/().%\s]+$")
AUTOMATION_LLM_PROFILE = "pro"
MAX_RUNTIME_LOOP_STEPS = 8


class AutomationService:
    def __init__(
        self,
        settings: Settings,
        llm: LlmService,
        artifact_service: ArtifactService | None = None,
        runtime: AgentRuntime | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.windows = WindowsAutomationService()
        self.openrpa = OpenRPAAdapter(settings)

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
        self._run_tasks: dict[str, asyncio.Task[Any]] = {}
        self._browser_lock = asyncio.Lock()
        self._playwright: Any = None
        self._browser_context: Any = None
        self._page: Any = None
        self.artifacts = artifact_service or ArtifactService(settings)
        self.runtime = runtime or AgentRuntime(settings, llm, self.artifacts)
        self.builder = AutomationBuilderService(self.runtime.registry, ALLOWED_AUTOMATION_TOOLS)
        self._runtime_context: dict[str, dict[str, Any]] = {}

    async def start_run(self, request: AutomationRunRequest) -> AutomationRun:
        recipe = self.get_recipe(request.recipe_id) if request.recipe_id else None
        prompt = recipe.prompt if recipe and not request.prompt.strip() else request.prompt.strip()
        run = AutomationRun(
            id=uuid.uuid4().hex,
            prompt=prompt,
            recipe_id=recipe.id if recipe else request.recipe_id,
            create_recipe=request.create_recipe,
            agent_state={"goal": prompt, "step_count": 0, "failure_history": [], "inputs": dict(request.inputs or {})},
        )
        self.runs[run.id] = run
        self._append_event(run, "queued", "Automation run queued.", {"prompt": run.prompt})
        self._save_run(run)
        self._run_tasks[run.id] = asyncio.create_task(self._execute_run(run.id, recipe))
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
        state = self._runtime_context_for(run)
        note = request.note.strip()
        if self._useful_continue_note(note):
            state["pending_input"] = note
        elif "pending_input" in state:
            state.pop("pending_input", None)
        if request.selected_artifact_id:
            state["selected_artifact_id"] = request.selected_artifact_id.strip()
        self._sync_runtime_context_to_run(run)

        event = self._continue_events.get(run_id)
        if event:
            event.set()
            self._set_status(run, "running")
            self._append_event(run, "continue", note or "User confirmed they are ready to continue.")
            self._save_run(run)
            return run

        if run.status == "waiting_for_user":
            self._set_status(run, "running")
            self._append_event(run, "continue", note or "Retrying with the current artifact context.", {"selected_artifact_id": request.selected_artifact_id or ""})
            self._run_tasks[run.id] = asyncio.create_task(self._execute_run(run.id, None, resume=True))
        else:
            self._append_event(run, "continue", note or "User confirmed they are ready to continue.")
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
        accepted = request.approved
        message = "Approved." if request.approved else "Cancelled by user."
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

    def engine_statuses(self) -> list[AutomationEngineStatus]:
        return [
            AutomationEngineStatus(
                id="astra",
                label="Astra Runtime",
                installed=True,
                configured_path="",
                message="Astra's built-in browser, artifact, and desktop runtime is available.",
            ),
            self.openrpa.status(),
        ]

    def get_recipe(self, recipe_id: str | None) -> AutomationRecipe | None:
        if not recipe_id:
            return None
        return next((recipe for recipe in self._read_recipes() if recipe.id == recipe_id), None)

    def create_recipe(self, request: AutomationRecipeCreateRequest) -> AutomationRecipe:
        recipes = [recipe for recipe in self._read_recipes() if recipe.name.strip().lower() != request.name.strip().lower()]
        if request.engine == "openrpa":
            recipe = self._create_openrpa_recipe(request)
            recipes.append(recipe)
            self._write_recipes(recipes)
            return recipe

        status = request.status
        steps = self._sanitize_steps(request.steps) if status == "executable" else self._sanitize_recipe_draft_steps(request.steps)
        if status == "executable" and len(steps) != len([step for step in request.steps if isinstance(step, dict)]):
            status = "draft"
        recipe = AutomationRecipe(
            id=uuid.uuid4().hex,
            name=request.name.strip(),
            prompt=request.prompt.strip(),
            engine="astra",
            steps=steps,
            inputs=self._dedupe_strings(request.inputs, limit=12),
            risk=request.risk,
            status=status,
            missing_tools=self._dedupe_strings(request.missing_tools, limit=24),
            validation_errors=self._dedupe_strings(request.validation_errors, limit=24),
            built_from=request.built_from.strip()[:500],
            aliases=self._dedupe_strings(request.aliases, limit=20),
            description=request.description.strip()[:1000],
        )
        recipes.append(recipe)
        self._write_recipes(recipes)
        return recipe

    def automation_suggestion(self, prompt: str) -> AutomationSuggestion | None:
        recipe = self.match_openrpa_recipe(prompt)
        if not recipe:
            return None
        return AutomationSuggestion(
            recipe=recipe,
            message=f"Matched registered OpenRPA workflow: {recipe.name}. Review it before running.",
            inputs=list(recipe.inputs),
            risk=recipe.risk,
        )

    def match_openrpa_recipe(self, prompt: str) -> AutomationRecipe | None:
        normalized = self._normalize_prompt(prompt)
        candidates: list[tuple[int, AutomationRecipe]] = []
        for recipe in self._read_recipes():
            if recipe.engine != "openrpa" or recipe.status != "executable" or recipe.risk == "blocked":
                continue
            aliases = [recipe.name, *recipe.aliases]
            score = 0
            for alias in aliases:
                alias_score = self._openrpa_alias_score(normalized, alias)
                score = max(score, alias_score)
            if score > 0:
                candidates.append((score, recipe))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return candidates[0][1]

    def _create_openrpa_recipe(self, request: AutomationRecipeCreateRequest) -> AutomationRecipe:
        workflow_ref = request.workflow_ref.strip()
        aliases = self._dedupe_strings(request.aliases, limit=20)
        inputs = self._dedupe_strings(request.inputs, limit=12)
        timeout_seconds = min(86400, max(1, int(request.timeout_seconds or 300)))
        recipe = AutomationRecipe(
            id=uuid.uuid4().hex,
            name=request.name.strip(),
            prompt=request.prompt.strip(),
            engine="openrpa",
            steps=[],
            inputs=inputs,
            risk=request.risk,
            status=request.status if request.status == "executable" else "executable",
            missing_tools=[],
            validation_errors=[],
            built_from=request.built_from.strip()[:500],
            workflow_ref=workflow_ref,
            workflow_ref_type=request.workflow_ref_type,
            aliases=aliases,
            timeout_seconds=timeout_seconds,
            description=request.description.strip()[:1000],
        )
        self._assert_openrpa_recipe_allowed(recipe)
        self.openrpa.validate_recipe(recipe)
        return recipe

    def _openrpa_alias_score(self, normalized_prompt: str, alias: str) -> int:
        normalized_alias = self._normalize_prompt(alias)
        if len(normalized_alias) < 3:
            return 0
        if normalized_alias in normalized_prompt:
            return 100 + len(normalized_alias)
        tokens = [token for token in re.split(r"[^a-z0-9]+", normalized_alias) if len(token) >= 3]
        if tokens and all(re.search(rf"\b{re.escape(token)}\b", normalized_prompt) for token in tokens):
            return 50 + len(tokens)
        return 0

    def _assert_openrpa_recipe_allowed(self, recipe: AutomationRecipe) -> None:
        if recipe.risk == "blocked":
            raise ValueError("This OpenRPA workflow is marked blocked and cannot be executed.")
        fields = [
            recipe.name,
            recipe.prompt,
            recipe.workflow_ref,
            recipe.description,
            *recipe.aliases,
            *recipe.inputs,
        ]
        joined = " ".join(str(item or "") for item in fields)
        if OPENRPA_UNSAFE_METADATA_PATTERN.search(joined):
            raise ValueError(
                "This OpenRPA workflow is blocked by Astra's safety policy. "
                "Do not register external workflows for credentials, payments, purchases, or destructive file actions."
            )

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

    def can_handle_agent_prompt(self, prompt: str) -> bool:
        normalized = self._normalize_prompt(prompt)
        if self.runtime.local_plan(prompt):
            return True
        if self._should_use_stable_plan(prompt):
            return True
        if re.search(r"\b(download|open|play|watch|listen|summari[sz]e|extract|search|find|browse)\b", normalized):
            return bool(self._extract_url(prompt) or self.artifacts.list_recent(limit=1))
        return False

    async def _execute_run(self, run_id: str, recipe: AutomationRecipe | None = None, resume: bool = False) -> None:
        run = self.get_run(run_id)
        if not run:
            return
        plan: list[dict[str, Any]] = []
        try:
            if recipe and recipe.status != "executable":
                missing = ", ".join(recipe.missing_tools) if recipe.missing_tools else "missing executable steps"
                raise ValueError(f"'{recipe.name}' is saved as {recipe.status}, not executable yet. Missing: {missing}.")

            if recipe and recipe.engine == "openrpa":
                await self._execute_openrpa_run(run, recipe)
                return

            if run.create_recipe and not recipe:
                await self._execute_recipe_builder_run(run, resume=resume)
                return

            if not recipe and self._should_use_runtime_loop(run.prompt, resume=resume):
                await self._execute_runtime_loop(run, resume=resume)
                return

            self._set_status(run, "planning")
            self._append_event(run, "planning", "Planning automation steps.")
            if BLOCKED_PROMPT_PATTERN.search(run.prompt) and not self._looks_like_destructive_windows_file_action(run.prompt):
                raise ValueError("This automation request is blocked by the safety policy.")

            plan = recipe.steps if recipe and recipe.steps else await self._plan_steps(run.prompt)
            plan = self._sanitize_steps(plan)
            if not plan:
                raise ValueError("Astra could not create runnable automation steps.")

            self._set_status(run, "running")
            self._append_event(run, "plan_ready", f"Prepared {len(plan)} automation steps.", {"steps": plan})
            for index, step in enumerate(plan):
                await self._execute_step(run, step)
                if run.status in {"cancelled", "error", "waiting_for_login", "waiting_for_user", "confirmation_required"}:
                    return
                if step["tool"] == "system.ask_user" and index == len(plan) - 1:
                    raise ValueError("Astra asked for input but did not have any executable follow-up steps, so it did not mark the task complete.")

            if run.create_recipe or any(step["tool"] == "automation.save_recipe" for step in plan):
                recipe = self._build_recipe_from_goal(run.prompt, plan, source_prompt=run.prompt)
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
                self._runtime_context.pop(run.id, None)
                self._run_tasks.pop(run.id, None)
                self._clear_active_download_process(run.id)
            elif run.status == "waiting_for_user":
                self._run_tasks.pop(run.id, None)

    def _should_use_runtime_loop(self, prompt: str, resume: bool = False) -> bool:
        if resume:
            return True
        if self._should_use_stable_plan(prompt):
            return False
        return bool(self.runtime.local_plan(prompt))

    async def _execute_recipe_builder_run(self, run: AutomationRun, resume: bool = False) -> None:
        self._set_status(run, "planning")
        self._append_event(run, "planning", "Planning reusable automation recipe.")
        state = self._runtime_context_for(run)
        clarification = str(state.get("pending_input") or "").strip()

        if self.builder.needs_goal(run.prompt) and not self._useful_continue_note(clarification):
            self._append_event(
                run,
                "plan_ready",
                "Prepared 1 recipe-builder step.",
                {"steps": [{"tool": "system.ask_user", "description": "Ask what automation to create.", "args": {}}]},
            )
            self._append_event(
                run,
                "step",
                "Ask what automation should be created.",
                {"step": {"tool": "system.ask_user", "description": "Ask what automation should be created.", "args": {}}},
            )
            await self._wait_for_user(
                run,
                "What kind of automation would you like to create? Describe the task or workflow to automate.",
                status="waiting_for_user",
            )
            state = self._runtime_context_for(run)
            clarification = str(state.get("pending_input") or "").strip()

        goal = self.builder.extract_goal(run.prompt, clarification)
        if not goal or self.builder.needs_goal(goal):
            raise ValueError("Astra needs a concrete automation goal before it can build a reusable workflow.")
        if BLOCKED_PROMPT_PATTERN.search(goal) and not self._looks_like_destructive_windows_file_action(goal):
            raise ValueError("This automation recipe is blocked by the safety policy.")

        planned_steps = await self._plan_steps(goal)
        recipe = self._build_recipe_from_goal(goal, planned_steps, source_prompt=run.prompt)
        run.recipe_id = recipe.id

        self._append_event(
            run,
            "recipe_saved",
            self._recipe_saved_message(recipe),
            {"recipe": recipe.model_dump(mode="json"), "status": recipe.status, "missing_tools": recipe.missing_tools},
        )
        run.result = self._recipe_saved_message(recipe)
        self._set_status(run, "complete")
        self._append_event(run, "complete", run.result, {"recipe_id": recipe.id, "recipe_status": recipe.status})

    def _build_recipe_from_goal(self, goal: str, planned_steps: list[dict[str, Any]] | None = None, source_prompt: str = "") -> AutomationRecipe:
        build = self.builder.build(goal, planned_steps=planned_steps, source_prompt=source_prompt)
        return self.create_recipe(build.request)

    def _recipe_saved_message(self, recipe: AutomationRecipe) -> str:
        if recipe.status == "executable":
            return f"Saved executable automation: {recipe.name}."
        if recipe.status == "needs_tools":
            missing = ", ".join(recipe.missing_tools) if recipe.missing_tools else "additional tools"
            return f"Saved draft automation: {recipe.name}. It needs tools before it can run: {missing}."
        return f"Saved draft automation: {recipe.name}."

    async def _execute_openrpa_run(self, run: AutomationRun, recipe: AutomationRecipe) -> None:
        self._set_status(run, "planning")
        self._append_event(
            run,
            "openrpa_queued",
            f"Preparing OpenRPA workflow: {recipe.name}.",
            {"engine": "openrpa", "recipe_id": recipe.id, "workflow_ref_type": recipe.workflow_ref_type},
        )
        self._assert_openrpa_recipe_allowed(recipe)

        status = self.openrpa.status()
        if not status.installed:
            raise OpenRPAMissingError(status.message)

        inputs = self._openrpa_run_inputs(run, recipe)
        missing_inputs = self._missing_openrpa_inputs(recipe, inputs)
        if missing_inputs:
            await self._wait_for_user(
                run,
                f"OpenRPA workflow '{recipe.name}' needs input: {', '.join(missing_inputs)}. Reply with key=value pairs.",
                status="waiting_for_user",
                event_type="openrpa_input_required",
            )
            self._raise_if_cancelled(run.id)
            note = str(self._runtime_context_for(run).get("pending_input") or "")
            inputs.update(self._parse_openrpa_continue_inputs(note, missing_inputs))
            state = self._runtime_context_for(run)
            state["inputs"] = dict(inputs)
            self._sync_runtime_context_to_run(run)
            missing_inputs = self._missing_openrpa_inputs(recipe, inputs)
            if missing_inputs:
                raise ValueError(f"OpenRPA workflow is missing required input: {', '.join(missing_inputs)}.")

        if recipe.risk == "safe_confirm":
            approved = await self._wait_for_confirmation(
                run,
                {
                    "tool": "openrpa.workflow",
                    "description": f"Launch OpenRPA workflow '{recipe.name}'.",
                    "args": {
                        "engine": "openrpa",
                        "workflow_ref": recipe.workflow_ref,
                        "workflow_ref_type": recipe.workflow_ref_type,
                        "inputs": list(inputs.keys()),
                    },
                },
            )
            if not approved:
                run.result = "OpenRPA workflow cancelled before launch."
                self._set_status(run, "cancelled")
                self._append_event(run, "cancelled", run.result)
                return

        self._set_status(run, "running")
        self._append_event(
            run,
            "openrpa_start",
            f"Launching OpenRPA workflow: {recipe.name}.",
            {
                "engine": "openrpa",
                "recipe_id": recipe.id,
                "workflow_ref": recipe.workflow_ref,
                "workflow_ref_type": recipe.workflow_ref_type,
                "inputs": list(inputs.keys()),
                "timeout_seconds": recipe.timeout_seconds,
            },
        )

        def on_openrpa_event(event_type: str, message: str, data: dict[str, Any]) -> None:
            self._append_event(
                run,
                f"openrpa_{event_type}",
                message,
                {"engine": "openrpa", "recipe_id": recipe.id, **(data or {})},
            )

        try:
            result = await asyncio.to_thread(
                self.openrpa.run,
                recipe,
                inputs,
                self._cancel_event_for_run(run.id),
                on_openrpa_event,
            )
        except OpenRPACancelledError as exc:
            raise AutomationCancelledError(str(exc)) from exc

        if result.returncode != 0:
            details = "\n".join(part for part in [result.stderr, result.stdout] if part).strip()
            raise ValueError(f"OpenRPA workflow failed with exit code {result.returncode}: {details[-1200:]}")

        output = result.stdout or result.stderr
        run.result = output[-3000:] if output else f"OpenRPA workflow '{recipe.name}' completed."
        self._set_status(run, "complete")
        self._append_event(
            run,
            "openrpa_complete",
            f"OpenRPA workflow completed: {recipe.name}.",
            {
                "engine": "openrpa",
                "recipe_id": recipe.id,
                "elapsed_seconds": round(result.elapsed_seconds, 2),
                "exit_code": result.returncode,
            },
        )
        self._append_event(run, "complete", run.result, {"engine": "openrpa", "recipe_id": recipe.id})

    def _openrpa_run_inputs(self, run: AutomationRun, recipe: AutomationRecipe) -> dict[str, Any]:
        state = self._runtime_context_for(run)
        raw_inputs = state.get("inputs") if isinstance(state.get("inputs"), dict) else {}
        allowed = set(recipe.inputs)
        return {str(key): value for key, value in dict(raw_inputs).items() if str(key) in allowed}

    def _missing_openrpa_inputs(self, recipe: AutomationRecipe, inputs: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for input_name in recipe.inputs:
            value = inputs.get(input_name)
            if value is None or str(value).strip() == "":
                missing.append(input_name)
        return missing

    def _parse_openrpa_continue_inputs(self, note: str, missing_inputs: list[str]) -> dict[str, str]:
        clean_note = note.strip()
        if not clean_note:
            return {}
        parsed_json = self._extract_json(clean_note)
        if isinstance(parsed_json, dict) and parsed_json:
            return {str(key).strip(): str(value).strip() for key, value in parsed_json.items() if str(key).strip()}

        pairs: dict[str, str] = {}
        for chunk in re.split(r"[\n,;]+", clean_note):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            key = key.strip()
            if key:
                pairs[key] = value.strip()
        if not pairs and len(missing_inputs) == 1:
            pairs[missing_inputs[0]] = clean_note
        return pairs

    async def _execute_runtime_loop(self, run: AutomationRun, resume: bool = False) -> None:
        state = self._runtime_context_for(run)
        state["goal"] = run.prompt
        state.setdefault("failure_history", [])
        if not resume:
            self._set_status(run, "planning")
            self._append_event(run, "planning", "Planning automation steps.")
        else:
            self._append_event(run, "replanned", "Replanning with the latest user input and artifact context.", {"agent_state": state})

        if BLOCKED_PROMPT_PATTERN.search(run.prompt) and not self._looks_like_destructive_windows_file_action(run.prompt):
            raise ValueError("This automation request is blocked by the safety policy.")

        self._set_status(run, "running")
        for _index in range(MAX_RUNTIME_LOOP_STEPS):
            self._raise_if_cancelled(run.id)
            state = self._runtime_context_for(run)
            state["step_count"] = int(state.get("step_count") or 0) + 1
            self._sync_runtime_context_to_run(run)
            action = await self.runtime.plan_next_action(run.prompt, run.agent_state, current_url=run.current_url, allow_llm=self._automation_llm_available())
            if not action:
                raise ValueError("Astra could not choose the next runtime action.")
            step = self._sanitize_steps([action])
            if not step:
                self._append_event(run, "replanned", "Planner returned an unsafe or unknown tool, so Astra replanned.", {"action": action})
                self._append_failure(run, action, "Planner returned an unsafe or unknown tool.")
                if len(self._runtime_context_for(run).get("failure_history", [])) >= 2:
                    raise ValueError("Astra could not create a safe next action.")
                continue

            next_step = step[0]
            event_type = "plan_ready" if int(state.get("step_count") or 0) == 1 and not resume else "replanned"
            self._append_event(run, event_type, f"Next action: {next_step['description']}", {"step": next_step, "agent_state": run.agent_state})
            result = await self._execute_runtime_step(run, next_step)
            self._record_runtime_result(run, next_step, result)

            if result["status"] == "complete":
                self._set_status(run, "complete")
                self._append_event(run, "complete", run.result or result["message"] or "Automation completed.", {"current_url": run.current_url})
                return
            if result["status"] == "needs_input":
                self._save_run(run)
                return
            if result["status"] == "failed":
                self._append_failure(run, next_step, result["message"])
                self._append_event(run, "replanned", "Astra will replan after the failed tool action.", {"failure": result, "step": next_step})
                if len(self._runtime_context_for(run).get("failure_history", [])) >= 2:
                    raise ValueError(result["message"] or "Runtime action failed.")
                continue

        raise ValueError("Astra reached the runtime step limit before completing the task.")

    async def _plan_steps(self, prompt: str) -> list[dict[str, Any]]:
        message_steps = self.builder.message_steps_for_goal(prompt)
        if message_steps:
            return message_steps

        use_stable_plan = self._should_use_stable_plan(prompt)
        if not use_stable_plan:
            try:
                runtime_steps = await self.runtime.plan(prompt, allow_llm=self._automation_llm_available())
                if runtime_steps:
                    return runtime_steps
            except Exception:
                pass
        if use_stable_plan:
            return self._heuristic_plan(prompt)
        return self._heuristic_plan(prompt)

    def _should_use_stable_plan(self, prompt: str) -> bool:
        normalized = prompt.lower()
        if self._looks_like_windows_prompt(prompt):
            return True
        if "youtube" in normalized or "you tube" in normalized or "yt " in normalized:
            return True
        if "gmail" in normalized or "mail.google" in normalized:
            return True
        return bool(self._extract_url(prompt))

    async def _plan_with_llm(self, prompt: str) -> list[dict[str, Any]]:
        return await self.runtime.plan(prompt, allow_llm=True)

    def _automation_planner_model(self) -> str:
        model_selector = getattr(self.llm, "model_for_profile", None)
        if callable(model_selector):
            return model_selector(AUTOMATION_LLM_PROFILE)
        return self.settings.resolved_cerebras_pro_model

    def _automation_llm_available(self) -> bool:
        model = self._automation_planner_model()
        provider = model.split(":", 1)[0].strip().lower() if ":" in model else "cerebras"
        provider_configured = getattr(self.llm, "provider_configured", None)
        if callable(provider_configured):
            return bool(provider_configured(provider))
        if provider == "nvidia":
            return self.settings.has_nvidia
        return self.settings.has_cerebras

    def _heuristic_plan(self, prompt: str) -> list[dict[str, Any]]:
        normalized = prompt.lower()
        steps: list[dict[str, Any]] = []
        create_recipe = "new automation" in normalized or "reusable automation" in normalized or "save" in normalized

        windows_steps = self._windows_heuristic_plan(prompt)
        if windows_steps:
            steps.extend(windows_steps)
            if create_recipe:
                steps.append({"tool": "automation.save_recipe", "description": "Save this workflow for reuse.", "args": {}})
            return steps

        if url := self._extract_url(prompt):
            if "download" in normalized:
                if self._looks_like_direct_file(url) and not self._is_youtube_url(url):
                    steps.append({"tool": "browser.download", "description": "Download from the direct URL.", "args": {"url": url}})
                else:
                    steps.append({"tool": "video.download_permitted", "description": "Download the provided public video URL.", "args": {"url": url}})
            else:
                steps.append({"tool": "browser.open", "description": f"Open {url}.", "args": {"url": url}})
                steps.append({"tool": "browser.extract", "description": "Extract page content.", "args": {"target": "page content"}})
        elif "gmail" in normalized or "mail.google" in normalized:
            steps.append({"tool": "browser.open", "description": "Open Gmail.", "args": {"url": "https://mail.google.com/"}})
            steps.append({"tool": "browser.extract", "description": "Read visible Gmail information.", "args": {"target": "first 5 emails"}})
        elif "youtube" in normalized or "you tube" in normalized or "yt " in normalized:
            youtube_intent = self._parse_youtube_intent(prompt)
            if youtube_intent["query"]:
                steps.append(
                    {
                        "tool": "youtube.search",
                        "description": "Search YouTube with matching filters.",
                        "args": youtube_intent,
                    }
                )
            else:
                steps.append({"tool": "browser.open", "description": "Open YouTube.", "args": {"url": "https://www.youtube.com/"}})
            if "download" in normalized:
                steps.append({"tool": "browser.click", "description": "Open the first visible YouTube result.", "args": {"selector": "ytd-video-renderer a#thumbnail"}})
                steps.append({"tool": "video.download_permitted", "description": "Download the opened public video.", "args": {}})
            elif youtube_intent["action"] == "play":
                steps.append({"tool": "youtube.result", "description": "Open the selected YouTube result.", "args": {"mode": "play", "index": 1}})
            elif youtube_intent["action"] == "name":
                steps.append({"tool": "youtube.result", "description": "Return the selected YouTube video title.", "args": {"mode": "name", "index": 1}})
            elif youtube_intent["query"]:
                steps.append({"tool": "youtube.result", "description": "Extract structured YouTube results.", "args": {"mode": "list", "index": 1}})
        elif "python" in normalized:
            steps.append({"tool": "python.run_safe", "description": "Run a constrained Python task.", "args": {"code": "print('Describe the Python automation steps more specifically.')"}})
        else:
            query = self._extract_search_query(prompt, ("open", "search", "find", "tell", "me", "about"))
            steps.append({"tool": "browser.search", "description": "Search the web.", "args": {"site": "google", "query": query or prompt}})
            steps.append({"tool": "browser.extract", "description": "Extract visible result information.", "args": {"target": "search results"}})

        if create_recipe:
            steps.append({"tool": "automation.save_recipe", "description": "Save this workflow for reuse.", "args": {}})
        return steps

    def _windows_heuristic_plan(self, prompt: str) -> list[dict[str, Any]]:
        normalized = self._normalize_prompt(prompt)
        if self._looks_like_destructive_windows_file_action(prompt):
            return [
                {
                    "tool": "windows.reject_destructive_file_action",
                    "description": "Reject destructive file automation.",
                    "args": {"reason": "Astra does not delete, move, rename, format, or wipe files in automation mode."},
                }
            ]

        if self._looks_like_alarm_intent(normalized):
            try:
                alarm_args = self._parse_alarm_args(prompt)
            except ValueError as exc:
                alarm_args = {"error": str(exc)}
            return [{"tool": "windows.set_alarm", "description": "Set a Windows alarm.", "args": alarm_args}]

        if self._looks_like_calculator_intent(normalized):
            expression = self._extract_calculator_expression(prompt)
            description = "Open Calculator."
            if expression:
                description = f"Open Calculator and enter {expression}."
            return [{"tool": "windows.open_calculator", "description": description, "args": {"expression": expression}}]

        if self._looks_like_file_explorer_intent(normalized):
            target = self._extract_file_explorer_target(prompt)
            description = f"Open File Explorer{f' to {target}' if target else ''}."
            return [{"tool": "windows.open_file_explorer", "description": description, "args": {"target": target}}]

        return []

    def _looks_like_windows_prompt(self, prompt: str) -> bool:
        normalized = self._normalize_prompt(prompt)
        return (
            self._looks_like_destructive_windows_file_action(prompt)
            or self._looks_like_alarm_intent(normalized)
            or self._looks_like_calculator_intent(normalized)
            or self._looks_like_file_explorer_intent(normalized)
        )

    def _looks_like_destructive_windows_file_action(self, prompt: str) -> bool:
        return bool(WINDOWS_DESTRUCTIVE_FILE_PATTERN.search(prompt))

    def _looks_like_alarm_intent(self, normalized: str) -> bool:
        return bool(re.search(r"\b(alarm|alarms|clock)\b", normalized) and re.search(r"\b(open|set|create|add|start)\b", normalized))

    def _looks_like_calculator_intent(self, normalized: str) -> bool:
        return bool(re.search(r"\b(calculator|calc)\b", normalized) and re.search(r"\b(open|launch|start|calculate|compute|what is|what's)\b", normalized))

    def _looks_like_file_explorer_intent(self, normalized: str) -> bool:
        if re.search(r"\b(file explorer|explorer)\b", normalized) and re.search(r"\b(open|launch|start|show)\b", normalized):
            return True
        return bool(re.search(r"\b(open|launch|start|show)\b", normalized) and re.search(r"\b(downloads?|documents?|desktop|pictures|videos|music)\s+(folder|directory)\b", normalized))

    def _parse_youtube_intent(self, prompt: str) -> dict[str, Any]:
        normalized = self._normalize_prompt(prompt)
        action = "results"
        if re.search(r"\b(play|watch|open\s+(?:the\s+)?(?:first|top|latest)|start)\b", normalized):
            action = "play"
        if re.search(r"\b(name|title|just\s+name|nothing\s+else|only\s+(?:the\s+)?(?:name|title))\b", normalized):
            action = "name"

        result_type = "video"
        if re.search(r"\bshorts?\b", normalized):
            result_type = "shorts"
        elif re.search(r"\blive\b", normalized):
            result_type = "live"
        elif re.search(r"\bchannels?\b", normalized) and not re.search(r"\bvideo|videos|latest|newest|recent|popular|play|watch|name|title\b", normalized):
            result_type = "channel"
        elif re.search(r"\bplaylists?\b", normalized):
            result_type = "playlist"
        elif not re.search(r"\bvideo|videos|latest|newest|recent|play|watch|name|title\b", normalized):
            result_type = "all"

        upload_date = ""
        if re.search(r"\blast\s+hour\b|\bpast\s+hour\b", normalized):
            upload_date = "hour"
        elif re.search(r"\btoday\b|\blast\s+24\s+hours?\b", normalized):
            upload_date = "today"
        elif re.search(r"\bthis\s+week\b|\bweek\b", normalized):
            upload_date = "week"
        elif re.search(r"\bthis\s+month\b|\bmonth\b", normalized):
            upload_date = "month"
        elif re.search(r"\bthis\s+year\b|\byear\b", normalized):
            upload_date = "year"
        elif re.search(r"\blatest|newest|recent|recently\s+uploaded\b", normalized):
            upload_date = "recent"

        duration = ""
        if re.search(r"\bshort\b|\bunder\s+4\s+minutes?\b", normalized):
            duration = "short"
        elif re.search(r"\blong\b|\bover\s+20\s+minutes?\b", normalized):
            duration = "long"
        elif re.search(r"\bmedium\b|\b4\s*-\s*20\s+minutes?\b", normalized):
            duration = "medium"

        sort = ""
        if re.search(r"\bmost\s+viewed|popular|views?\b", normalized):
            sort = "view_count"
        elif re.search(r"\brating|top\s+rated\b", normalized):
            sort = "rating"
        elif upload_date in {"hour", "today", "week", "month", "year", "recent"}:
            sort = "upload_date"

        source = self._extract_youtube_source_and_content(prompt)
        query = self._extract_youtube_query(prompt)
        channel_hint = self._has_youtube_channel_hint(normalized)
        topic_hint = self._has_youtube_topic_hint(normalized, query)
        route = "auto"
        if source.get("content_query"):
            route = "topic"
        elif channel_hint:
            route = "channel"
        elif topic_hint:
            route = "topic"
        return {
            "query": query,
            "action": action,
            "result_type": result_type,
            "upload_date": upload_date,
            "duration": duration,
            "sort": sort,
            "features": [],
            "route": route,
            "channel_hint": channel_hint,
            "channel_query": source.get("channel_query") or "",
            "content_query": source.get("content_query") or "",
            "source_qualified": bool(source.get("content_query")),
        }

    def _extract_youtube_query(self, prompt: str) -> str:
        quoted = re.search(r"['\"]([^'\"]{1,240})['\"]", prompt)
        if quoted:
            return re.sub(r"\s+", " ", quoted.group(1)).strip()

        channel_url_query = self._extract_youtube_channel_url_query(prompt)
        if channel_url_query:
            return channel_url_query

        cleaned = re.sub(r"https?://[^\s]+", " ", prompt, flags=re.IGNORECASE)
        source = self._extract_youtube_source_and_content(cleaned)
        if source.get("content_query"):
            return re.sub(r"\s+", " ", f"{source['content_query']} {source['channel_query']}").strip()[:240]
        channel_query = self._extract_youtube_channel_query(cleaned)
        if channel_query:
            return channel_query
        cleaned = cleaned.replace("it's", " ").replace("itâ€™s", " ")
        phrases = (
            "you tube",
            "recently uploaded",
            "last 24 hours",
            "last hour",
            "past hour",
            "this week",
            "this month",
            "this year",
            "just name",
            "nothing else",
            "only the name",
            "only name",
            "only the title",
            "sort by",
            "filter by",
            "upload date",
            "view count",
        )
        for phrase in phrases:
            cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b(youtube|yt|open|go|to|search|find|look|up|for|play|watch|start|download|latest|newest|recent|today|week|month|year|"
            r"video|videos|short|shorts|live|channel|channels|playlist|playlists|filter|filters|name|title|the|a|an|and|or|on|in|"
            r"first|top|result|results|please|only|with|under|over|minutes?|uploaded|by|rating|popular|view|views|count|else|about|stream|streams)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        query = re.sub(r"[^a-zA-Z0-9 .'_+-]+", " ", cleaned)
        query = re.sub(r"\s+", " ", query).strip(" .")
        return query[:240]

    def _extract_youtube_source_and_content(self, prompt: str) -> dict[str, str]:
        cleaned = re.sub(r"https?://[^\s]+", " ", prompt, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        patterns = (
            r"\b(?:from|by|uploaded\s+by|creator)\s+(.+?)\s*(?:[:;|]|\s+-\s+)\s*(.+)$",
            r"\b(?:from|by|uploaded\s+by|creator)\s+(.+?)\s+(?:about|for|on)\s+(.+)$",
            r"\b(?:from|by|uploaded\s+by|creator)\s+(.+?)\s+(?:song|track|music|video)\s+(?:called|named|titled)\s+(.+)$",
            r"\bplay\s+(.+?)(?:\s+(?:song|track|music|video))?\s+(?:from|by)\s+(.+)$",
        )
        for index, pattern in enumerate(patterns):
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if not match:
                continue
            if index == 3:
                content_raw, channel_raw = match.group(1), match.group(2)
            else:
                channel_raw, content_raw = match.group(1), match.group(2)
            channel = self._clean_youtube_query_fragment(channel_raw)
            content = self._clean_youtube_query_fragment(content_raw)
            if channel and content:
                return {"channel_query": channel[:120], "content_query": content[:240]}
        return {"channel_query": "", "content_query": ""}

    def _extract_youtube_channel_query(self, prompt: str) -> str:
        channel_url_query = self._extract_youtube_channel_url_query(prompt)
        if channel_url_query:
            return channel_url_query

        channel_patterns = (
            r"\b(?:from|by|uploaded\s+by|creator)\s+(.+?)(?:\s+(?:channel|on\s+youtube|youtube\s+channel))?(?:\s+(?:and|just|only|nothing|please|with|for|to|latest|newest|recent|popular|most\s+viewed|video|videos|short|shorts|live|stream|streams|play|watch|name|title)\b|$)",
            r"\bsearch\s+(?:for\s+)?(.+?)\s+(?:youtube\s+)?channel\b",
            r"\bopen\s+(?:the\s+)?(.+?)\s+(?:youtube\s+)?channel\b",
            r"\b(.+?)\s+(?:youtube\s+)?channel\s+(?:latest|newest|recent|popular|most\s+viewed|video|videos|short|shorts|live|stream|streams|play|watch|name|title)\b",
        )
        for pattern in channel_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                candidate = self._clean_youtube_query_fragment(match.group(1))
                if candidate:
                    return candidate[:240]
        return ""

    def _extract_youtube_channel_url_query(self, prompt: str) -> str:
        match = re.search(
            r"(?:https?://)?(?:www\.)?youtube\.com/(?:(?:@|c/|user/)([^/\s?#]+)|channel/([^/\s?#]+))",
            prompt,
            re.IGNORECASE,
        )
        if not match:
            return ""
        value = match.group(1) or match.group(2) or ""
        value = value.lstrip("@")
        return self._clean_youtube_query_fragment(value)[:240]

    def _clean_youtube_query_fragment(self, value: str) -> str:
        cleaned = re.sub(r"https?://[^\s]+", " ", value, flags=re.IGNORECASE)
        cleaned = cleaned.replace("it's", " ")
        cleaned = re.sub(
            r"\b(youtube|yt|open|go|to|search|find|look|up|for|play|watch|start|download|latest|newest|recent|today|week|month|year|"
            r"video|videos|short|shorts|live|channel|channels|playlist|playlists|filter|filters|name|title|the|a|an|and|or|on|in|"
            r"first|top|result|results|please|just|nothing|only|with|under|over|minutes?|uploaded|rating|popular|view|views|count|else|from|by)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[^a-zA-Z0-9 .'_+-]+", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip(" .")

    def _has_youtube_channel_hint(self, normalized_prompt: str) -> bool:
        return bool(
            re.search(r"\b(?:youtube\s+)?channel\b", normalized_prompt)
            or re.search(r"(?:https?://)?(?:www\.)?youtube\.com/(?:@|channel/|c/|user/)[^\s]+", normalized_prompt)
            or re.search(r"\b(?:from|by|uploaded\s+by|creator)\s+[\w@.'+-]", normalized_prompt)
        )

    def _has_youtube_topic_hint(self, normalized_prompt: str, query: str) -> bool:
        haystack = f"{normalized_prompt} {query.lower()}"
        return bool(
            re.search(
                r"\b(tutorial|how\s+to|learn|course|programming|coding|guide|explained|explanation|review|comparison|"
                r"vs|versus|news\s+about|about|recipe|workout|lecture|class|lesson|documentary|highlights?|compilation)\b",
                haystack,
            )
        )

    def _parse_alarm_args(self, prompt: str) -> dict[str, Any]:
        target_time = self._parse_alarm_time(prompt)
        label = self._extract_alarm_label(prompt)
        return {
            "target_iso": target_time.isoformat(),
            "display_time": target_time.strftime("%I:%M %p").lstrip("0"),
            "label": label,
        }

    def _parse_alarm_time(self, prompt: str) -> datetime:
        now = self._now()
        normalized = self._normalize_prompt(prompt)
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", normalized)
        if not match:
            raise ValueError("Tell Astra the alarm time, for example 5:30 PM today.")

        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        meridiem = match.group(3)
        if minute > 59:
            raise ValueError("Alarm minutes must be between 00 and 59.")
        if meridiem:
            if hour < 1 or hour > 12:
                raise ValueError("Use a 1-12 hour value with AM or PM.")
            if meridiem == "pm" and hour != 12:
                hour += 12
            if meridiem == "am" and hour == 12:
                hour = 0
        elif hour > 23:
            raise ValueError("Use a valid hour for the alarm.")

        target_date = now.date()
        explicit_today = "today" in normalized
        explicit_tomorrow = "tomorrow" in normalized
        if explicit_tomorrow:
            target_date = (now + timedelta(days=1)).date()

        target_time = datetime.combine(target_date, datetime.min.time()).replace(hour=hour, minute=minute)
        if target_time <= now:
            if explicit_today:
                raise ValueError("That alarm time has already passed today, so Astra did not set an alarm. Choose a future time such as tomorrow.")
            target_time += timedelta(days=1)
        return target_time

    def _extract_alarm_label(self, prompt: str) -> str:
        match = re.search(r"\b(?:called|named|label(?:ed)?)\s+(.+)$", prompt, re.IGNORECASE)
        if not match:
            return "Astra alarm"
        label = re.sub(r"\s+", " ", match.group(1)).strip(" .")
        return label[:80] or "Astra alarm"

    def _extract_calculator_expression(self, prompt: str) -> str:
        cleaned = re.sub(r"\b(open|launch|start|calculator|calc|calculate|compute|what is|what's|please|and|enter|type)\b", " ", prompt, flags=re.IGNORECASE)
        cleaned = cleaned.replace("x", "*").replace("X", "*").replace("÷", "/").replace("×", "*")
        expression = re.sub(r"\s+", "", cleaned)
        if not expression or not CALCULATOR_EXPRESSION_PATTERN.fullmatch(expression):
            return ""
        return expression[:80]

    def _extract_file_explorer_target(self, prompt: str) -> str:
        normalized = self._normalize_prompt(prompt)
        drive_match = re.search(r"\b([a-z]):\\?\b", prompt, re.IGNORECASE)
        if drive_match:
            return f"{drive_match.group(1).upper()}:\\"
        drive_word_match = re.search(r"\b([a-z])\s+drive\b", normalized)
        if drive_word_match:
            return f"{drive_word_match.group(1).upper()}:\\"
        for target in ("downloads", "documents", "desktop", "pictures", "videos", "music"):
            if re.search(rf"\b{target}\b", normalized):
                return target
        path_match = re.search(r"([a-zA-Z]:\\[^\n\r]+)", prompt)
        if path_match:
            return path_match.group(1).strip().strip("\"'")
        return ""

    def _normalize_prompt(self, prompt: str) -> str:
        return re.sub(r"\s+", " ", prompt.lower()).strip()

    def _useful_continue_note(self, note: str) -> bool:
        compact = re.sub(r"[^a-z0-9]+", " ", note.lower()).strip()
        return bool(compact and compact not in {"continue", "ready", "user is ready to continue", "ok", "okay", "yes", "y"})

    def _now(self) -> datetime:
        return datetime.now()

    def _sanitize_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        clean_steps: list[dict[str, Any]] = []
        for raw in steps:
            if not isinstance(raw, dict):
                continue
            tool = str(raw.get("tool") or "").strip()
            tool = {"video.download": "video.download_permitted", "automation.run_python_safe": "python.run_safe"}.get(tool, tool)
            if tool not in ALLOWED_AUTOMATION_TOOLS:
                continue
            args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
            description = str(raw.get("description") or tool).strip()[:240]
            if tool in {"artifact.resolve_reference", "file.search_scoped"}:
                args["query"] = str(args.get("query") or "").strip()[:240]
                media_types = args.get("media_types")
                args["media_types"] = [str(item).strip().lower() for item in media_types[:6]] if isinstance(media_types, list) else []
                if not args["query"]:
                    continue
            if tool == "artifact.list_recent":
                media_types = args.get("media_types")
                args["media_types"] = [str(item).strip().lower() for item in media_types[:6]] if isinstance(media_types, list) else []
                try:
                    args["limit"] = min(25, max(1, int(args.get("limit") or 10)))
                except (TypeError, ValueError):
                    args["limit"] = 10
            if tool == "artifact.pick":
                args["artifact_id"] = str(args.get("artifact_id") or "").strip()[:80]
                if not args["artifact_id"]:
                    continue
            if tool == "app.resolve":
                args["app_name"] = str(args.get("app_name") or "default").strip()[:80]
            if tool == "app.open":
                args["app_name"] = str(args.get("app_name") or "").strip()[:80]
                if not args["app_name"]:
                    continue
            if tool in {"app.open_with_file", "file.open", "artifact.open_containing_folder"}:
                args["artifact_id"] = str(args.get("artifact_id") or "").strip()[:80]
                args["reference"] = str(args.get("reference") or args.get("query") or "").strip()[:240]
                args["app_name"] = str(args.get("app_name") or "default").strip()[:80]
                args["path"] = str(args.get("path") or "").strip()[:500]
            if tool in {"desktop.find_text", "desktop.verify_text"}:
                args["text"] = str(args.get("text") or "").strip()[:240]
                args["app_name"] = str(args.get("app_name") or "").strip()[:80]
                args["control_types"] = self._sanitize_string_list(args.get("control_types"), limit=8)
                args["exclude_control_types"] = self._sanitize_string_list(args.get("exclude_control_types"), limit=8)
                try:
                    args["timeout"] = min(20, max(1, int(args.get("timeout") or 8)))
                except (TypeError, ValueError):
                    args["timeout"] = 8
                if not args["text"]:
                    continue
            if tool == "desktop.click":
                target = args.get("target")
                args["target"] = target if isinstance(target, dict) else str(target or "").strip()[:240]
                args["text"] = str(args.get("text") or "").strip()[:240]
                args["button"] = str(args.get("button") or "left").strip().lower()[:20]
                if args["button"] not in {"left", "right"}:
                    args["button"] = "left"
                if not args["target"] and not args["text"]:
                    continue
            if tool == "desktop.type_text":
                args["text"] = str(args.get("text") or "").strip()[:1000]
                args["replace"] = bool(args.get("replace"))
                if not args["text"]:
                    continue
            if tool == "desktop.press_key":
                key = str(args.get("key") or "").strip().lower()
                key = {"return": "enter", "esc": "escape"}.get(key, key)
                if key not in {"enter", "tab", "escape", "backspace", "delete", "space"}:
                    continue
                args["key"] = key
            if tool == "system.ask_user":
                args["message"] = str(args.get("message") or description).strip()[:300]
            if tool == "task.finish":
                args["message"] = str(args.get("message") or description).strip()[:500]
            if tool == "task.replan":
                args["reason"] = str(args.get("reason") or description).strip()[:300]
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
            if tool == "youtube.search":
                args["query"] = str(args.get("query") or "").strip()[:240]
                args["action"] = str(args.get("action") or "results").strip().lower()[:20]
                args["result_type"] = str(args.get("result_type") or "video").strip().lower()[:20]
                args["upload_date"] = str(args.get("upload_date") or "").strip().lower()[:20]
                args["duration"] = str(args.get("duration") or "").strip().lower()[:20]
                args["sort"] = str(args.get("sort") or "").strip().lower()[:20]
                args["route"] = str(args.get("route") or "auto").strip().lower()[:20]
                args["channel_hint"] = bool(args.get("channel_hint"))
                args["channel_query"] = str(args.get("channel_query") or "").strip()[:120]
                args["content_query"] = str(args.get("content_query") or "").strip()[:240]
                args["source_qualified"] = bool(args.get("source_qualified"))
                args["features"] = self._sanitize_string_list(args.get("features"), limit=8)
                for key, allowed, default in (
                    ("action", {"results", "name", "play"}, "results"),
                    ("result_type", {"all", "video", "shorts", "live", "channel", "playlist"}, "video"),
                    ("upload_date", {"", "hour", "today", "week", "month", "year", "recent"}, ""),
                    ("duration", {"", "short", "medium", "long"}, ""),
                    ("sort", {"", "relevance", "upload_date", "view_count", "rating"}, ""),
                    ("route", {"auto", "channel", "topic"}, "auto"),
                ):
                    if args[key] not in allowed:
                        args[key] = default
                if not args["query"]:
                    continue
            if tool == "youtube.result":
                mode = str(args.get("mode") or "list").strip().lower()
                args["mode"] = mode if mode in {"list", "name", "play"} else "list"
                try:
                    args["index"] = min(10, max(1, int(args.get("index") or 1)))
                except (TypeError, ValueError):
                    args["index"] = 1
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
            if tool == "windows.set_alarm":
                args["target_iso"] = str(args.get("target_iso") or "").strip()
                args["display_time"] = str(args.get("display_time") or "").strip()[:40]
                args["label"] = str(args.get("label") or "Astra alarm").strip()[:80]
                args["error"] = str(args.get("error") or "").strip()[:240]
            if tool == "windows.open_calculator":
                expression = str(args.get("expression") or "").strip().replace(" ", "")
                args["expression"] = expression[:80] if expression and CALCULATOR_EXPRESSION_PATTERN.fullmatch(expression) else ""
            if tool == "windows.open_file_explorer":
                args["target"] = str(args.get("target") or "").strip()[:260]
            if tool == "windows.prepare_whatsapp_message":
                args["contact"] = str(args.get("contact") or "").strip()[:120]
                args["message"] = str(args.get("message") or "").strip()[:500]
                if not args["contact"] or not args["message"]:
                    continue
            if tool == "windows.send_prepared_whatsapp_message":
                args["contact"] = str(args.get("contact") or "").strip()[:120]
                args["expected_message"] = str(args.get("expected_message") or args.get("message") or "").strip()[:500]
                if not args["expected_message"]:
                    continue
            if tool == "windows.reject_destructive_file_action":
                args["reason"] = str(args.get("reason") or "Astra blocked this destructive file action.").strip()[:240]
            clean_steps.append({"tool": tool, "description": description, "args": args})
        return clean_steps

    def _sanitize_recipe_draft_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed_draft_tools = ALLOWED_AUTOMATION_TOOLS | FUTURE_DESKTOP_TOOLS | {"app.open"}
        clean_steps: list[dict[str, Any]] = []
        for raw in steps:
            if not isinstance(raw, dict):
                continue
            tool = str(raw.get("tool") or "").strip()
            tool = {"video.download": "video.download_permitted", "automation.run_python_safe": "python.run_safe"}.get(tool, tool)
            if tool not in allowed_draft_tools:
                continue
            args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
            clean_args = {str(key)[:80]: self._safe_recipe_arg(value) for key, value in args.items() if str(key).strip()}
            clean_steps.append(
                {
                    "tool": tool,
                    "description": str(raw.get("description") or tool).strip()[:240],
                    "args": clean_args,
                }
            )
        return clean_steps

    def _safe_recipe_arg(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()[:1000]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, list):
            return [self._safe_recipe_arg(item) for item in value[:20]]
        if isinstance(value, dict):
            return {str(key)[:80]: self._safe_recipe_arg(item) for key, item in list(value.items())[:20]}
        return str(value)[:500]

    def _dedupe_strings(self, values: list[str], limit: int = 20) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = str(value or "").strip()[:160]
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                result.append(clean)
            if len(result) >= limit:
                break
        return result

    def _sanitize_string_list(self, value: Any, limit: int = 10) -> list[str]:
        if not isinstance(value, list):
            return []
        return self._dedupe_strings([str(item) for item in value], limit=limit)

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
        elif tool == "youtube.search":
            await self._tool_youtube_search(run, step["args"])
        elif tool == "youtube.result":
            await self._tool_youtube_result(run, step["args"])
        elif tool in {"artifact.resolve_reference", "file.search_scoped"}:
            await self._tool_artifact_resolve_reference(run, step["args"])
        elif tool == "artifact.list_recent":
            await self._tool_artifact_list_recent(run, step["args"])
        elif tool == "artifact.pick":
            await self._tool_artifact_pick(run, step["args"])
        elif tool == "artifact.open_containing_folder":
            await self._tool_artifact_open_containing_folder(run, step["args"])
        elif tool == "app.resolve":
            await self._tool_app_resolve(run, step["args"])
        elif tool == "app.open":
            await self._tool_app_open(run, step["args"])
        elif tool in {"app.open_with_file", "file.open"}:
            await self._tool_app_open_with_file(run, step["args"])
        elif tool == "desktop.find_text":
            await self._tool_desktop_find_text(run, step["args"])
        elif tool == "desktop.click":
            await self._tool_desktop_click(run, step["args"])
        elif tool == "desktop.type_text":
            await self._tool_desktop_type_text(run, step["args"])
        elif tool == "desktop.press_key":
            await self._tool_desktop_press_key(run, step["args"])
        elif tool == "desktop.verify_text":
            await self._tool_desktop_verify_text(run, step["args"])
        elif tool == "python.run_safe":
            await self._tool_python_run_safe(run, step["args"]["code"])
        elif tool == "video.download_permitted":
            await self._tool_video_download_permitted(run, step["args"])
        elif tool == "system.ask_user":
            await self._wait_for_user(run, str(step["args"].get("message") or step["description"]), status="waiting_for_user")
        elif tool == "windows.set_alarm":
            await self._tool_windows_set_alarm(run, step["args"])
        elif tool == "windows.open_calculator":
            await self._tool_windows_open_calculator(run, step["args"])
        elif tool == "windows.open_file_explorer":
            await self._tool_windows_open_file_explorer(run, step["args"])
        elif tool == "windows.prepare_whatsapp_message":
            await self._tool_windows_prepare_whatsapp_message(run, step["args"])
        elif tool == "windows.send_prepared_whatsapp_message":
            await self._tool_windows_send_prepared_whatsapp_message(run, step["args"])
        elif tool == "windows.reject_destructive_file_action":
            await self._tool_windows_reject_destructive_file_action(run, step["args"])
        elif tool == "task.finish":
            run.result = str(step["args"].get("message") or run.result or "Automation completed.").strip()
            self._append_event(run, "verified", run.result, {"step": step})
        elif tool == "task.replan":
            self._append_event(run, "replanned", str(step["args"].get("reason") or "Replanning automation."), {"step": step})

    async def _execute_runtime_step(self, run: AutomationRun, step: dict[str, Any]) -> dict[str, Any]:
        try:
            before_status = run.status
            before_event_count = len(run.events)
            await self._execute_step(run, step)
            if run.status in {"waiting_for_user", "waiting_for_login"}:
                return {"status": "needs_input", "message": run.result or run.events[-1].message if run.events else "Astra needs input.", "data": run.agent_state}
            if run.status in {"cancelled", "error"}:
                return {"status": "failed", "message": run.error or run.result or f"Runtime step ended as {run.status}.", "data": {}}
            if step["tool"] == "task.finish":
                return {"status": "complete", "message": run.result or "Automation completed.", "data": {}}
            new_events = run.events[before_event_count:]
            message = new_events[-1].message if new_events else run.result or step["description"]
            return {"status": "success", "message": message, "data": {"previous_status": before_status, "events": [event.model_dump(mode="json") for event in new_events]}}
        except AutomationCancelledError:
            raise
        except Exception as exc:
            return {"status": "failed", "message": self._format_exception(exc), "data": {"step": step}}

    def _record_runtime_result(self, run: AutomationRun, step: dict[str, Any], result: dict[str, Any]) -> None:
        state = self._runtime_context_for(run)
        state["last_tool"] = step["tool"]
        state["last_tool_status"] = result["status"]
        state["last_result"] = result["message"]
        self._sync_runtime_context_to_run(run)

    def _append_failure(self, run: AutomationRun, step_or_action: dict[str, Any], message: str) -> None:
        state = self._runtime_context_for(run)
        failures = state.get("failure_history")
        if not isinstance(failures, list):
            failures = []
        failures.append({"tool": step_or_action.get("tool", ""), "message": message, "at": datetime.utcnow().isoformat()})
        state["failure_history"] = failures[-5:]
        self._sync_runtime_context_to_run(run)

    async def _tool_windows_set_alarm(self, run: AutomationRun, args: dict[str, Any]) -> None:
        if args.get("error"):
            message = str(args["error"])
            run.result = message
            self._append_event(run, "blocked", message, {"app": "Windows Clock", "reason": "invalid_alarm_time"})
            self._set_status(run, "cancelled")
            return

        target_iso = str(args.get("target_iso") or "").strip()
        if not target_iso:
            raise ValueError("Alarm automation needs a target time.")
        try:
            target_time = datetime.fromisoformat(target_iso)
        except ValueError as exc:
            raise ValueError("Alarm automation received an invalid target time.") from exc
        if target_time <= self._now():
            message = "That alarm time has already passed, so Astra did not set an alarm. Choose a future time."
            run.result = message
            self._append_event(run, "blocked", message, {"app": "Windows Clock", "reason": "past_alarm_time", "alarm_time": target_iso})
            self._set_status(run, "cancelled")
            return

        label = str(args.get("label") or "Astra alarm").strip()[:80]
        result = await asyncio.to_thread(self.windows.set_alarm, target_time, label)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

    async def _tool_windows_open_calculator(self, run: AutomationRun, args: dict[str, Any]) -> None:
        expression = str(args.get("expression") or "").strip()
        result = await asyncio.to_thread(self.windows.open_calculator, expression)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

    async def _tool_windows_open_file_explorer(self, run: AutomationRun, args: dict[str, Any]) -> None:
        target = str(args.get("target") or "").strip()
        result = await asyncio.to_thread(self.windows.open_file_explorer, target)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

    async def _tool_windows_prepare_whatsapp_message(self, run: AutomationRun, args: dict[str, Any]) -> None:
        contact = str(args.get("contact") or "").strip()
        message = str(args.get("message") or "").strip()
        result = await asyncio.to_thread(self.windows.prepare_whatsapp_message, contact, message)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        context = self._runtime_context_for(run)
        context["last_app_name"] = "WhatsApp"
        context["whatsapp_contact"] = contact
        context["whatsapp_message_length"] = len(message)
        self._sync_runtime_context_to_run(run)
        run.result = result.message

    async def _tool_windows_send_prepared_whatsapp_message(self, run: AutomationRun, args: dict[str, Any]) -> None:
        contact = str(args.get("contact") or "").strip()
        expected_message = str(args.get("expected_message") or args.get("message") or "").strip()
        result = await asyncio.to_thread(self.windows.send_prepared_whatsapp_message, contact, expected_message)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        context = self._runtime_context_for(run)
        context["last_app_name"] = "WhatsApp"
        context["whatsapp_contact"] = contact or context.get("whatsapp_contact", "")
        context["whatsapp_sent_message_length"] = len(expected_message)
        self._sync_runtime_context_to_run(run)
        run.result = result.message

    async def _tool_windows_reject_destructive_file_action(self, run: AutomationRun, args: dict[str, Any]) -> None:
        message = str(args.get("reason") or "Astra blocked this destructive file automation.").strip()
        run.result = message
        self._append_event(run, "blocked", message, {"tool": "windows.reject_destructive_file_action"})
        self._set_status(run, "cancelled")

    def _append_windows_result(self, run: AutomationRun, result: WindowsAutomationResult) -> None:
        data = {"ok": result.ok, **result.data}
        self._append_event(run, result.event_type, result.message, data)

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

    async def _tool_youtube_search(self, run: AutomationRun, args: dict[str, Any]) -> None:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("YouTube search needs a query.")

        route = str(args.get("route") or "auto").strip().lower()
        if route not in {"auto", "channel", "topic"}:
            route = "auto"
        context = self._runtime_context_for(run)
        if route in {"auto", "channel"}:
            channel_match = await self._search_youtube_channel_videos(run, args, explicit=route == "channel")
            if channel_match:
                channel, results = channel_match
                filters = self._youtube_filter_context(args)
                context["youtube_query"] = query
                context["youtube_filters"] = filters
                context["youtube_route"] = "channel"
                context["youtube_channel"] = channel
                context["youtube_results"] = results
                self._sync_runtime_context_to_run(run)
                summary = self._format_youtube_results(results)
                run.result = summary
                self._append_event(
                    run,
                    "youtube_channel_search",
                    f"Opened {channel.get('title') or query} channel and found {len(results)} YouTube result{'s' if len(results) != 1 else ''}.",
                    {
                        "query": query,
                        "filters": filters,
                        "channel": channel,
                        "results": results[:5],
                        "url": run.current_url,
                    },
                )
                return
            if route == "channel":
                raise ValueError(f"No matching YouTube channel videos were found for {query}.")

        page = await self._ensure_page()
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        run.current_url = page.url
        await self._wait_for_youtube_ready(page)
        await self._apply_youtube_filters(page, args)
        await self._wait_for_youtube_ready(page)
        results = await self._extract_youtube_results(page, str(args.get("result_type") or "video"))
        if bool(args.get("source_qualified")):
            results = self._rank_youtube_source_results(args, results)

        filters = self._youtube_filter_context(args)
        context["youtube_query"] = query
        context["youtube_filters"] = filters
        context["youtube_route"] = "topic"
        context.pop("youtube_channel", None)
        context["youtube_results"] = results
        self._sync_runtime_context_to_run(run)

        if not results:
            raise ValueError(f"No YouTube results were found for {query}.")
        summary = self._format_youtube_results(results)
        run.result = summary
        self._append_event(
            run,
            "youtube_search",
            f"Found {len(results)} YouTube result{'s' if len(results) != 1 else ''} for {query}.",
            {"query": query, "filters": filters, "results": results[:5], "url": page.url},
        )

    def _rank_youtube_source_results(self, args: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        source = self._normalize_youtube_match_text(str(args.get("channel_query") or ""))
        content = self._normalize_youtube_match_text(str(args.get("content_query") or ""))
        if not source or not content:
            return results

        content_tokens = set(content.split())
        source_tokens = set(source.split())

        def overlap_score(tokens: set[str], text: str) -> float:
            if not tokens:
                return 0.0
            text_tokens = set(self._normalize_youtube_match_text(text).split())
            return len(tokens & text_tokens) / len(tokens)

        def score(item: dict[str, Any]) -> float:
            title = str(item.get("title") or "")
            channel = str(item.get("channel") or "")
            metadata = str(item.get("metadata") or "")
            title_norm = self._normalize_youtube_match_text(title)
            channel_norm = self._normalize_youtube_match_text(channel)
            source_text = f"{channel_norm} {title_norm}"
            source_score = overlap_score(source_tokens, source_text)
            if source and source in source_text:
                source_score = max(source_score, 1.0)
            content_score = overlap_score(content_tokens, f"{title} {metadata}")
            if content and content in title_norm:
                content_score = max(content_score, 1.0)
            return (source_score * 0.55) + (content_score * 0.4) + (0.05 if source_score and content_score else 0.0)

        return sorted(results, key=score, reverse=True)

    def _youtube_filter_context(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "result_type": args.get("result_type") or "video",
            "upload_date": args.get("upload_date") or "",
            "duration": args.get("duration") or "",
            "sort": args.get("sort") or "",
            "features": args.get("features") or [],
            "route": args.get("route") or "auto",
            "channel_query": args.get("channel_query") or "",
            "content_query": args.get("content_query") or "",
            "source_qualified": bool(args.get("source_qualified")),
        }

    async def _search_youtube_channel_videos(
        self,
        run: AutomationRun,
        args: dict[str, Any],
        *,
        explicit: bool,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        query = str(args.get("query") or "").strip()
        if not query:
            return None
        if not explicit and self._has_youtube_topic_hint(self._normalize_prompt(query), query):
            return None

        page = await self._ensure_page()
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
        run.current_url = page.url
        await self._wait_for_youtube_ready(page)
        await self._click_youtube_filter(page, "Channel")
        await self._wait_for_youtube_ready(page)

        candidates = await self._extract_youtube_channel_candidates(page)
        scored = sorted(
            (
                (self._score_youtube_channel_candidate(query, candidate), candidate)
                for candidate in candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        threshold = 0.42 if explicit else 0.82
        if not scored or scored[0][0] < threshold:
            return None

        channel = dict(scored[0][1])
        channel["score"] = round(scored[0][0], 3)
        section_url = self._youtube_channel_section_url(str(channel.get("url") or ""), args)
        if not self._valid_http_url(section_url):
            return None

        await page.goto(section_url, wait_until="domcontentloaded", timeout=45000)
        run.current_url = page.url
        await self._wait_for_youtube_channel_ready(page)
        await self._apply_youtube_channel_sort(page, args)
        await self._wait_for_youtube_channel_ready(page)

        result_type = str(args.get("result_type") or "video").lower()
        if result_type not in {"shorts", "live"}:
            result_type = "video"
        results = await self._extract_youtube_results(page, result_type)
        if not results and result_type == "live":
            results = await self._extract_youtube_results(page, "video")
        return (channel, results) if results else None

    async def _extract_youtube_channel_candidates(self, page: Any) -> list[dict[str, Any]]:
        candidates = await page.evaluate(
            """
            () => {
              const clean = (value) => (value || "").replace(/\\s+/g, " ").trim();
              const absolute = (href) => {
                try { return new URL(href, location.origin).toString(); } catch { return ""; }
              };
              const out = [];
              const seen = new Set();
              const push = (item) => {
                item.title = clean(item.title);
                item.url = absolute(item.url);
                item.handle = clean(item.handle);
                item.metadata = clean(item.metadata);
                if (!item.title || !item.url) return;
                let parsed;
                try { parsed = new URL(item.url); } catch { return; }
                if (!/(^|\\.)youtube\\.com$/.test(parsed.hostname)) return;
                if (!/^\\/(?:@|channel\\/|c\\/|user\\/)/.test(parsed.pathname)) return;
                const key = parsed.pathname.replace(/\\/+$/, "").toLowerCase();
                if (seen.has(key)) return;
                seen.add(key);
                out.push(item);
              };

              for (const card of document.querySelectorAll("ytd-channel-renderer, ytd-compact-channel-renderer, ytd-grid-channel-renderer")) {
                const anchor = card.querySelector("a#main-link, a[href^='/@'], a[href^='/channel/'], a[href^='/c/'], a[href^='/user/']");
                const title = card.querySelector("#channel-title, #text, yt-formatted-string")?.textContent || anchor?.textContent || "";
                const handle = card.querySelector("#subscribers, #metadata, #video-count")?.textContent || "";
                push({ title, url: anchor?.href || anchor?.getAttribute("href") || "", handle, metadata: card.innerText || "" });
              }

              for (const anchor of document.querySelectorAll("a[href^='/@'], a[href^='/channel/'], a[href^='/c/'], a[href^='/user/']")) {
                const href = anchor.getAttribute("href") || "";
                if (/\\/(feed|results|shorts|watch|playlist)(\\/|$)/.test(href)) continue;
                const card = anchor.closest("ytd-channel-renderer, ytd-compact-channel-renderer, ytd-grid-channel-renderer") || anchor.parentElement;
                push({
                  title: anchor.getAttribute("title") || anchor.getAttribute("aria-label") || anchor.textContent,
                  url: anchor.href || href,
                  handle: href.split("/").filter(Boolean).pop() || "",
                  metadata: card?.innerText || "",
                });
              }
              return out.slice(0, 12);
            }
            """
        )
        if not isinstance(candidates, list):
            return []
        clean_candidates: list[dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            title = self._clean_youtube_channel_title(str(item.get("title") or ""))[:160]
            url = str(item.get("url") or "").strip()
            if not title or not self._valid_http_url(url):
                continue
            clean_candidates.append(
                {
                    "title": title,
                    "url": url,
                    "handle": self._compact_text(str(item.get("handle") or ""))[:120],
                    "metadata": self._compact_text(str(item.get("metadata") or ""))[:400],
                }
            )
        return clean_candidates

    def _clean_youtube_channel_title(self, title: str) -> str:
        clean = self._compact_text(title)
        clean = re.sub(r"\s+@[\w.-]+.*$", "", clean)
        words = clean.split()
        if len(words) >= 2 and len(words) % 2 == 0:
            half = len(words) // 2
            if " ".join(words[:half]).lower() == " ".join(words[half:]).lower():
                clean = " ".join(words[:half])
        return clean.strip()

    def _score_youtube_channel_candidate(self, query: str, candidate: dict[str, Any]) -> float:
        query_norm = self._normalize_youtube_match_text(query)
        title_norm = self._normalize_youtube_match_text(str(candidate.get("title") or ""))
        handle_norm = self._normalize_youtube_match_text(str(candidate.get("handle") or ""))
        path_handle = self._normalize_youtube_match_text(urlparse(str(candidate.get("url") or "")).path.rsplit("/", 1)[-1].lstrip("@"))
        if not query_norm or not title_norm:
            return 0.0
        query_compact = query_norm.replace(" ", "")
        title_compact = title_norm.replace(" ", "")
        handle_compact = handle_norm.replace(" ", "")
        path_compact = path_handle.replace(" ", "")
        if query_norm in {title_norm, handle_norm, path_handle}:
            return 1.0
        if query_compact and query_compact in {title_compact, handle_compact, path_compact}:
            return 1.0
        if title_norm.startswith(query_norm) or path_handle.startswith(query_norm):
            return 0.94
        if query_compact and (title_compact.startswith(query_compact) or path_compact.startswith(query_compact)):
            return 0.94
        if query_norm in title_norm or query_norm in path_handle:
            return 0.88
        if query_compact and (query_compact in title_compact or query_compact in path_compact):
            return 0.88

        query_tokens = set(query_norm.split())
        candidate_tokens = set(title_norm.split()) | set(handle_norm.split()) | set(path_handle.split())
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens)
        recall = overlap / len(query_tokens)
        precision = overlap / len(candidate_tokens)
        if recall <= 0:
            return 0.0
        return max(0.0, min(0.84, (recall * 0.72) + (precision * 0.18)))

    def _normalize_youtube_match_text(self, value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
        normalized = re.sub(r"\b(official|youtube|channel|videos?)\b", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def _youtube_channel_section_url(self, channel_url: str, args: dict[str, Any]) -> str:
        if not self._valid_http_url(channel_url):
            return ""
        parsed = urlparse(channel_url)
        path = re.sub(r"/(?:videos|shorts|streams|featured|about|community)/?$", "", parsed.path.rstrip("/"), flags=re.IGNORECASE)
        if not path:
            path = parsed.path.rstrip("/")
        result_type = str(args.get("result_type") or "video").lower()
        section = "videos"
        if result_type == "shorts":
            section = "shorts"
        elif result_type == "live":
            section = "streams"
        return f"{parsed.scheme}://{parsed.netloc}{path}/{section}"

    async def _wait_for_youtube_channel_ready(self, page: Any) -> None:
        try:
            await page.wait_for_selector("ytd-rich-grid-media, ytd-rich-item-renderer, a#video-title-link, a[href*='/watch'], a[href*='/shorts/']", timeout=18000)
        except Exception:
            await page.wait_for_timeout(1500)

    async def _apply_youtube_channel_sort(self, page: Any, args: dict[str, Any]) -> None:
        sort = str(args.get("sort") or "").lower()
        upload_date = str(args.get("upload_date") or "").lower()
        if sort == "view_count":
            await self._click_youtube_chip(page, "Popular")
        elif sort == "upload_date" or upload_date in {"hour", "today", "week", "month", "year", "recent"}:
            if not await self._click_youtube_chip(page, "Latest"):
                await self._click_youtube_chip(page, "Recently uploaded")

    async def _tool_youtube_result(self, run: AutomationRun, args: dict[str, Any]) -> None:
        mode = str(args.get("mode") or "list").strip().lower()
        index = max(1, int(args.get("index") or 1))
        context = self._runtime_context_for(run)
        results = context.get("youtube_results") if isinstance(context.get("youtube_results"), list) else []
        if not results:
            page = await self._ensure_page()
            results = await self._extract_youtube_results(page, "video")
            context["youtube_results"] = results
            self._sync_runtime_context_to_run(run)
        if not results:
            raise ValueError("No YouTube video results are available.")

        selected = results[min(index - 1, len(results) - 1)]
        title = str(selected.get("title") or "").strip()
        url = str(selected.get("url") or "").strip()
        if not title:
            raise ValueError("The selected YouTube result did not expose a title.")

        if mode == "name":
            run.result = title
            self._append_event(run, "youtube_title", title, {"result": selected, "index": index})
            return

        if mode == "play":
            if not self._valid_http_url(url):
                raise ValueError("The selected YouTube result did not expose a playable URL.")
            page = await self._ensure_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=45000)
            run.current_url = page.url
            run.result = f"Playing: {title}"
            self._append_event(run, "youtube_playing", f"Playing: {title}", {"result": selected, "url": page.url, "index": index})
            return

        summary = self._format_youtube_results(results)
        run.result = summary
        self._append_event(run, "youtube_results", summary, {"results": results[:10], "index": index})

    async def _wait_for_youtube_ready(self, page: Any) -> None:
        try:
            await page.wait_for_selector("ytd-video-renderer, ytd-channel-renderer, ytd-playlist-renderer, ytd-reel-shelf-renderer, ytd-rich-grid-media, a#video-title, a#video-title-link, a[href*='/shorts/']", timeout=18000)
        except Exception:
            await page.wait_for_timeout(1200)

    async def _apply_youtube_filters(self, page: Any, args: dict[str, Any]) -> None:
        result_type = str(args.get("result_type") or "video").lower()
        upload_date = str(args.get("upload_date") or "").lower()
        duration = str(args.get("duration") or "").lower()
        sort = str(args.get("sort") or "").lower()

        if result_type == "shorts":
            await self._click_youtube_chip(page, "Shorts")
        elif result_type == "live":
            await self._click_youtube_chip(page, "Live")
        elif result_type == "video":
            await self._click_youtube_chip(page, "Videos")
        elif result_type == "channel":
            await self._click_youtube_filter(page, "Channel")
        elif result_type == "playlist":
            await self._click_youtube_filter(page, "Playlist")

        if upload_date == "recent" or sort == "upload_date":
            if not await self._click_youtube_chip(page, "Recently uploaded"):
                await self._click_youtube_filter(page, "Upload date")
        upload_labels = {
            "hour": "Last hour",
            "today": "Today",
            "week": "This week",
            "month": "This month",
            "year": "This year",
        }
        if upload_date in upload_labels:
            await self._click_youtube_filter(page, upload_labels[upload_date])

        duration_labels = {"short": "Under 4 minutes", "medium": "4 - 20 minutes", "long": "Over 20 minutes"}
        if duration in duration_labels:
            await self._click_youtube_filter(page, duration_labels[duration])

        sort_labels = {"view_count": "View count", "rating": "Rating"}
        if sort in sort_labels:
            await self._click_youtube_filter(page, sort_labels[sort])

    async def _click_youtube_chip(self, page: Any, label: str) -> bool:
        patterns = [
            page.get_by_role("button", name=re.compile(rf"^{re.escape(label)}$", re.IGNORECASE)),
            page.locator("yt-chip-cloud-chip-renderer").filter(has_text=re.compile(rf"^{re.escape(label)}$", re.IGNORECASE)),
            page.get_by_text(label, exact=True),
        ]
        for locator in patterns:
            try:
                await locator.first.click(timeout=2500)
                await page.wait_for_timeout(900)
                return True
            except Exception:
                continue
        return False

    async def _click_youtube_filter(self, page: Any, label: str) -> bool:
        try:
            await page.get_by_role("button", name=re.compile(r"filters?", re.IGNORECASE)).first.click(timeout=3000)
            await page.wait_for_timeout(400)
        except Exception:
            pass
        variants = [label]
        if label.lower() == "channel":
            variants.append("Channels")
        elif label.lower() == "playlist":
            variants.append("Playlists")
        for variant in variants:
            pattern = re.compile(rf"^{re.escape(variant)}$", re.IGNORECASE)
            for locator in (
                page.locator("ytd-search-filter-renderer").filter(has_text=pattern).locator("a"),
                page.locator("ytd-search-filter-options-dialog-renderer").get_by_text(pattern),
                page.get_by_role("link", name=pattern),
                page.get_by_text(variant, exact=True),
            ):
                try:
                    await locator.first.click(timeout=3000)
                    await page.wait_for_timeout(1200)
                    return True
                except Exception:
                    try:
                        await locator.first.click(timeout=1500, force=True)
                        await page.wait_for_timeout(1200)
                        return True
                    except Exception:
                        continue
        return False

    async def _extract_youtube_results(self, page: Any, result_type: str = "video") -> list[dict[str, Any]]:
        desired_type = (result_type or "video").lower()
        results = await page.evaluate(
            """
            (desiredType) => {
              const clean = (value) => (value || "").replace(/\\s+/g, " ").trim();
              const absolute = (href) => {
                try { return new URL(href, location.origin).toString(); } catch { return ""; }
              };
              const seen = new Set();
              const out = [];
              const push = (item) => {
                item.title = clean(item.title);
                item.url = absolute(item.url);
                item.channel = clean(item.channel);
                item.metadata = clean(item.metadata);
                if (!item.title || !item.url) return;
                const titleKey = item.title.toLowerCase();
                if (["shorts", "live", "upcoming", "course", "playlist", "view full course"].includes(titleKey) || titleKey.includes("now playing") || /^\\d{1,2}:\\d{2}(?::\\d{2})?(?:\\s+\\d{1,2}:\\d{2})?$/.test(titleKey)) return;
                try {
                  const parsedUrl = new URL(item.url, location.origin);
                  if (parsedUrl.pathname.replace(/\\/+$/, "") === "/shorts") return;
                } catch {}
                let key = item.url;
                try {
                  const parsedForKey = new URL(item.url, location.origin);
                  if (item.type === "playlist") key = `playlist:${parsedForKey.searchParams.get("list") || parsedForKey.pathname + parsedForKey.search}`;
                } catch {}
                if (seen.has(key)) return;
                seen.add(key);
                out.push(item);
              };

              for (const card of document.querySelectorAll("ytd-channel-renderer, ytd-compact-channel-renderer, ytd-grid-channel-renderer")) {
                const anchor = card.querySelector("a#main-link, a[href^='/@'], a[href^='/channel/'], a[href^='/c/'], a[href^='/user/']");
                const title = card.querySelector("#channel-title, #text, yt-formatted-string")?.textContent || anchor?.getAttribute("title") || anchor?.textContent || "";
                const metadata = card.querySelector("#metadata, #subscribers, #video-count")?.innerText || card.innerText || "";
                push({
                  title,
                  url: anchor?.href || anchor?.getAttribute("href") || "",
                  channel: "",
                  metadata,
                  type: "channel",
                });
              }

              if (desiredType === "channel") {
                for (const anchor of document.querySelectorAll("a[href^='/@'], a[href^='/channel/'], a[href^='/c/'], a[href^='/user/']")) {
                  const href = anchor.getAttribute("href") || "";
                  if (/\\/(feed|results|shorts|watch|playlist)(\\/|$)/.test(href)) continue;
                  const card = anchor.closest("ytd-channel-renderer, ytd-compact-channel-renderer, ytd-grid-channel-renderer, ytd-video-renderer, ytd-rich-item-renderer") || anchor.parentElement;
                  const aria = clean(anchor.getAttribute("aria-label") || "").replace(/^go to channel\\s+/i, "");
                  const title = anchor.getAttribute("title") || anchor.textContent || aria;
                  push({
                    title,
                    url: anchor.href || href,
                    channel: "",
                    metadata: card?.innerText || "",
                    type: "channel",
                  });
                }
              }

              for (const card of document.querySelectorAll("ytd-playlist-renderer, ytd-radio-renderer")) {
                const anchor = card.querySelector("a[href*='list='], a#video-title, h3 a");
                const title = card.querySelector("#playlist-title, h3, .yt-lockup-metadata-view-model__title, #video-title")?.textContent || anchor?.getAttribute("title") || anchor?.textContent || "";
                const owner = card.querySelector("ytd-channel-name a, #channel-name a, .yt-lockup-metadata-view-model__metadata")?.textContent || "";
                push({
                  title,
                  url: anchor?.href || anchor?.getAttribute("href") || "",
                  channel: owner,
                  metadata: card.innerText || "",
                  type: "playlist",
                });
              }

              for (const anchor of document.querySelectorAll("a[href*='/playlist?list='], a[href*='list=']")) {
                const card = anchor.closest("ytd-playlist-renderer, ytd-radio-renderer, ytd-rich-item-renderer, ytd-video-renderer, ytd-rich-grid-media, ytd-item-section-renderer") || anchor.parentElement;
                const title = card?.querySelector("#playlist-title, h3, .yt-lockup-metadata-view-model__title, #video-title")?.textContent || anchor.getAttribute("title") || anchor.getAttribute("aria-label") || anchor.textContent || "";
                push({
                  title,
                  url: anchor.href || anchor.getAttribute("href") || "",
                  channel: card?.querySelector("ytd-channel-name a, #channel-name a")?.textContent || "",
                  metadata: card?.innerText || "",
                  type: "playlist",
                });
              }

              const videoAnchors = Array.from(document.querySelectorAll("a#video-title[href*='/watch'], a#video-title-link[href*='/watch'], ytd-video-renderer a[href*='/watch'], ytd-rich-item-renderer a[href*='/watch']"));
              for (const anchor of videoAnchors) {
                const card = anchor.closest("ytd-video-renderer, ytd-rich-grid-media, ytd-rich-item-renderer, ytd-compact-video-renderer") || anchor.parentElement;
                const text = clean(card?.innerText || "");
                const badges = text.toLowerCase();
                const isLive = /\\blive\\b|watching now/.test(badges);
                push({
                  title: anchor.getAttribute("title") || card?.querySelector("#video-title, #video-title-link, h3, .yt-lockup-metadata-view-model__title")?.textContent || anchor.getAttribute("aria-label") || anchor.textContent,
                  url: anchor.href,
                  channel: card?.querySelector("ytd-channel-name a, #channel-name a")?.textContent || "",
                  metadata: card?.querySelector("#metadata-line")?.innerText || text.split("\\n").slice(1, 5).join(" "),
                  type: isLive ? "live" : "video",
                });
              }

              const shortAnchors = Array.from(document.querySelectorAll("a[href*='/shorts/']"));
              for (const anchor of shortAnchors) {
                const card = anchor.closest("ytd-reel-item-renderer, ytd-reel-video-renderer, ytd-rich-item-renderer, ytd-video-renderer") || anchor.parentElement;
                push({
                  title: anchor.getAttribute("title") || anchor.getAttribute("aria-label") || card?.querySelector("#video-title, .yt-lockup-metadata-view-model__title")?.textContent || card?.innerText?.split("\\n")?.[0],
                  url: anchor.href,
                  channel: card?.querySelector("ytd-channel-name a, #channel-name a")?.textContent || "",
                  metadata: card?.innerText || "",
                  type: "shorts",
                });
              }

              return out
                .filter((item) => {
                  if (desiredType === "all") return true;
                  if (desiredType === "shorts") return item.type === "shorts";
                  if (desiredType === "live") return item.type === "live";
                  if (desiredType === "video") return item.type === "video";
                  if (desiredType === "channel") return item.type === "channel";
                  if (desiredType === "playlist") return item.type === "playlist";
                  return item.type === "video";
                })
                .slice(0, 12);
            }
            """,
            desired_type,
        )
        if not isinstance(results, list):
            return []
        clean_results: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "video").strip()[:20]
            title = self._clean_youtube_title(str(item.get("title") or "").strip(), item_type)
            if item_type == "channel":
                title = self._clean_youtube_channel_title(title)
            url = str(item.get("url") or "").strip()
            if not title or self._bad_youtube_title(title) or not self._valid_http_url(url):
                continue
            title_key = f"{item_type}:{self._normalize_youtube_match_text(title)}"
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            metadata_limit = 220 if item_type in {"channel", "playlist"} else 500
            clean_results.append(
                {
                    "title": title[:300],
                    "url": url,
                    "channel": str(item.get("channel") or "").strip()[:120],
                    "metadata": self._compact_text(str(item.get("metadata") or ""))[:metadata_limit],
                    "type": item_type,
                }
            )
        return clean_results

    def _clean_youtube_title(self, title: str, result_type: str = "video") -> str:
        clean = re.sub(r"\s+", " ", title or "").strip()
        clean = re.sub(r"\s+\d+\s+hours?,\s+\d+\s+minutes?\s*$", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+\d+(?:\.\d+)?\s+(?:seconds?|minutes?|hours?)\s*,?\s*$", "", clean, flags=re.IGNORECASE)
        if result_type == "shorts":
            clean = re.sub(r"^New(?=[A-Z0-9])", "", clean)
            clean = re.sub(r"\s*\d+(?:\.\d+)?[KMB]?\s*views?\s*$", "", clean, flags=re.IGNORECASE)
        return clean.strip()

    def _bad_youtube_title(self, title: str) -> bool:
        clean = re.sub(r"\s+", " ", title or "").strip().lower()
        return bool(clean in {"shorts", "live", "upcoming", "course", "playlist", "view full course"} or "now playing" in clean or re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s+\d{1,2}:\d{2})?", clean))

    def _format_youtube_results(self, results: list[dict[str, Any]]) -> str:
        lines = []
        for index, item in enumerate(results[:5], start=1):
            detail = " - ".join(part for part in [str(item.get("channel") or "").strip(), str(item.get("metadata") or "").strip()] if part)
            suffix = f" ({detail})" if detail else ""
            lines.append(f"{index}. {item.get('title')}{suffix}")
        return "YouTube results:\n" + "\n".join(lines)

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
            artifact = self._record_download_artifact(run, target, source_url=url, title=target.stem)
            self._append_event(run, "download", run.result, {"path": str(target), "filename": target.name, "artifact": artifact.model_dump(mode="json")})
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
        artifact = self._record_download_artifact(run, target, source_url=page.url if page else "", title=target.stem)
        self._append_event(run, "download", run.result, {"path": str(target), "filename": target.name, "artifact": artifact.model_dump(mode="json")})

    async def _tool_artifact_resolve_reference(self, run: AutomationRun, args: dict[str, Any]) -> None:
        state = self._runtime_context_for(run)
        selected_artifact_id = str(args.get("artifact_id") or state.get("selected_artifact_id") or "").strip()
        if selected_artifact_id:
            await self._tool_artifact_pick(run, {"artifact_id": selected_artifact_id})
            return
        query = str(args.get("query") or state.get("pending_input") or run.prompt).strip()
        if not self._useful_continue_note(query):
            query = run.prompt
        media_types = args.get("media_types") if isinstance(args.get("media_types"), list) else []
        preferred_artifact_id = str(state.get("last_artifact_id") or "").strip()
        artifact, matches, reason = self.artifacts.resolve_reference(query, media_types=media_types, preferred_artifact_id=preferred_artifact_id)
        if not artifact:
            choices = [
                {"id": item.id, "title": item.title, "filename": item.filename, "media_type": item.media_type, "path": item.path}
                for item in matches[:5]
            ]
            message = reason or "Astra could not resolve which artifact to use."
            if choices:
                message = f"{message} Matching files: " + ", ".join(item["filename"] for item in choices)
            run.result = message
            state["candidates"] = choices
            state["pending_reason"] = message
            self._sync_runtime_context_to_run(run)
            self._append_event(run, "waiting_for_user", message, {"query": query, "matches": choices, "candidates": choices})
            self._set_status(run, "waiting_for_user")
            return

        state["last_artifact_id"] = artifact.id
        state.pop("selected_artifact_id", None)
        state.pop("pending_input", None)
        state.pop("pending_reason", None)
        state["candidates"] = []
        self._sync_runtime_context_to_run(run)
        self._append_event(
            run,
            "artifact_resolved",
            f"Resolved {artifact.filename}.",
            {
                "artifact": artifact.model_dump(mode="json"),
                "artifact_id": artifact.id,
                "filename": artifact.filename,
                "path": artifact.path,
                "matches": [item.model_dump(mode="json") for item in matches[:5]],
            },
        )

    async def _tool_artifact_list_recent(self, run: AutomationRun, args: dict[str, Any]) -> None:
        media_types = args.get("media_types") if isinstance(args.get("media_types"), list) else []
        limit = int(args.get("limit") or 10)
        artifacts = self.artifacts.list_recent(media_types=media_types, limit=limit)
        run.result = "Recent artifacts: " + ", ".join(item.filename for item in artifacts) if artifacts else "No Astra artifacts found yet."
        self._append_event(
            run,
            "artifact_list",
            run.result,
            {"artifacts": [artifact.model_dump(mode="json") for artifact in artifacts]},
        )

    async def _tool_artifact_pick(self, run: AutomationRun, args: dict[str, Any]) -> None:
        artifact_id = str(args.get("artifact_id") or "").strip()
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            raise ValueError("The selected artifact is no longer available.")
        state = self._runtime_context_for(run)
        state["last_artifact_id"] = artifact.id
        state.pop("selected_artifact_id", None)
        state.pop("pending_input", None)
        state["candidates"] = []
        self._sync_runtime_context_to_run(run)
        self._append_event(
            run,
            "artifact_resolved",
            f"Selected {artifact.filename}.",
            {"artifact": artifact.model_dump(mode="json"), "artifact_id": artifact.id, "filename": artifact.filename, "path": artifact.path},
        )

    async def _tool_artifact_open_containing_folder(self, run: AutomationRun, args: dict[str, Any]) -> None:
        artifact = self._artifact_from_args_or_context(run, args)
        if not artifact:
            raise ValueError("Astra could not resolve which artifact folder to open.")
        result = await asyncio.to_thread(self.runtime.open_artifact_containing_folder, artifact)
        state = self._runtime_context_for(run)
        state["last_artifact_id"] = artifact.id
        run.result = f"Opened the folder containing {artifact.filename}."
        self._sync_runtime_context_to_run(run)
        self._append_event(
            run,
            "folder_opened",
            run.result,
            {
                **result,
                "artifact": artifact.model_dump(mode="json"),
                "filename": artifact.filename,
                "path": artifact.path,
            },
        )

    async def _tool_app_resolve(self, run: AutomationRun, args: dict[str, Any]) -> None:
        app_name = str(args.get("app_name") or "default")
        app = self.runtime.resolve_app(app_name)
        context = self._runtime_context_for(run)
        context["last_app_name"] = str(app.get("app_name") or app_name)
        context["last_app_path"] = str(app.get("path") or "")
        self._sync_runtime_context_to_run(run)
        self._append_event(run, "app_resolved", f"Resolved app: {context['last_app_name']}.", {"app": app})

    async def _tool_app_open(self, run: AutomationRun, args: dict[str, Any]) -> None:
        app_name = str(args.get("app_name") or self._runtime_context_for(run).get("last_app_name") or "").strip()
        if not app_name:
            raise ValueError("Desktop app open step needs an app name.")
        result = await asyncio.to_thread(self.windows.open_app, app_name)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        context = self._runtime_context_for(run)
        context["last_app_name"] = str(result.data.get("app") or app_name)
        self._sync_runtime_context_to_run(run)
        run.result = result.message

    async def _tool_app_open_with_file(self, run: AutomationRun, args: dict[str, Any]) -> None:
        context = self._runtime_context_for(run)
        artifact = self._artifact_from_args_or_context(run, args)
        if not artifact:
            raise ValueError("Astra could not resolve which artifact to open.")
        app_name = str(args.get("app_name") or context.get("last_app_name") or "default").strip() or "default"
        result = await asyncio.to_thread(self.runtime.open_artifact_with_app, artifact, app_name)
        context["last_artifact_id"] = artifact.id
        context["last_app_name"] = str(result.get("app", {}).get("app_name") or app_name)
        run.result = f"Opened {artifact.filename} with {context['last_app_name']}."
        self._sync_runtime_context_to_run(run)
        self._append_event(
            run,
            "app_opened",
            run.result,
            {
                **result,
                "filename": artifact.filename,
                "path": artifact.path,
                "artifact": artifact.model_dump(mode="json"),
            },
        )

    async def _tool_desktop_find_text(self, run: AutomationRun, args: dict[str, Any]) -> None:
        text = str(args.get("text") or "").strip()
        app_name = str(args.get("app_name") or self._runtime_context_for(run).get("last_app_name") or "").strip()
        timeout = int(args.get("timeout") or 8)
        result = await asyncio.to_thread(
            self.windows.find_text,
            text,
            app_name,
            timeout,
            args.get("control_types") if isinstance(args.get("control_types"), list) else [],
            args.get("exclude_control_types") if isinstance(args.get("exclude_control_types"), list) else [],
        )
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        target = result.data.get("target") if isinstance(result.data, dict) else None
        if isinstance(target, dict):
            context = self._runtime_context_for(run)
            context["last_desktop_target"] = target
            context["last_desktop_text"] = text
            self._sync_runtime_context_to_run(run)
        run.result = result.message

    async def _tool_desktop_click(self, run: AutomationRun, args: dict[str, Any]) -> None:
        context = self._runtime_context_for(run)
        target: Any = args.get("target")
        if target == "$found_text":
            target = context.get("last_desktop_target")
        text = str(args.get("text") or "").strip()
        button = str(args.get("button") or "left").strip().lower()
        result = await asyncio.to_thread(self.windows.click_target, target, text, button)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

    async def _tool_desktop_type_text(self, run: AutomationRun, args: dict[str, Any]) -> None:
        text = str(args.get("text") or "")
        result = await asyncio.to_thread(self.windows.type_text, text, bool(args.get("replace")))
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

    async def _tool_desktop_press_key(self, run: AutomationRun, args: dict[str, Any]) -> None:
        key = str(args.get("key") or "").strip().lower()
        result = await asyncio.to_thread(self.windows.press_key, key)
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

    async def _tool_desktop_verify_text(self, run: AutomationRun, args: dict[str, Any]) -> None:
        text = str(args.get("text") or "").strip()
        app_name = str(args.get("app_name") or self._runtime_context_for(run).get("last_app_name") or "").strip()
        result = await asyncio.to_thread(
            self.windows.verify_text,
            text,
            app_name,
            int(args.get("timeout") or 6),
            args.get("control_types") if isinstance(args.get("control_types"), list) else [],
            args.get("exclude_control_types") if isinstance(args.get("exclude_control_types"), list) else [],
        )
        self._append_windows_result(run, result)
        if not result.ok:
            raise ValueError(result.message)
        run.result = result.message

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
        artifact = self._record_download_artifact(
            run,
            result["path"],
            source_url=source_url,
            title=str(result.get("video_title") or result["filename"]),
            metadata=result,
        )
        result = {**result, "artifact": artifact.model_dump(mode="json"), "artifact_id": artifact.id}
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

    def _runtime_context_for(self, run: AutomationRun) -> dict[str, Any]:
        if not isinstance(run.agent_state, dict):
            run.agent_state = {}
        context = self._runtime_context.setdefault(run.id, {})
        for key, value in run.agent_state.items():
            context.setdefault(key, value)
        return context

    def _sync_runtime_context_to_run(self, run: AutomationRun) -> None:
        context = self._runtime_context.get(run.id)
        if context is not None:
            run.agent_state = dict(context)

    def _artifact_from_args_or_context(self, run: AutomationRun, args: dict[str, Any]) -> AutomationArtifact | None:
        artifact_id = str(args.get("artifact_id") or "").strip()
        if artifact_id:
            return self.artifacts.get(artifact_id)

        path = str(args.get("path") or "").strip()
        if path:
            return self.artifacts.get_by_path(path)

        reference = str(args.get("reference") or "").strip()
        if reference:
            artifact, _matches, _reason = self.artifacts.resolve_reference(reference)
            if artifact:
                return artifact

        context_artifact_id = str(self._runtime_context_for(run).get("last_artifact_id") or "").strip()
        if context_artifact_id:
            return self.artifacts.get(context_artifact_id)
        return None

    def _record_download_artifact(
        self,
        run: AutomationRun,
        path: str | Path,
        source_url: str = "",
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AutomationArtifact:
        artifact = self.artifacts.record_download(path, run.id, source_url=source_url, title=title, metadata=metadata)
        self._runtime_context_for(run)["last_artifact_id"] = artifact.id
        self._sync_runtime_context_to_run(run)
        self._append_event(
            run,
            "artifact_recorded",
            f"Remembered {artifact.filename} for future tasks.",
            {"artifact": artifact.model_dump(mode="json"), "artifact_id": artifact.id, "filename": artifact.filename, "path": artifact.path},
        )
        return artifact

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

    async def _wait_for_user(self, run: AutomationRun, reason: str, status: str = "waiting_for_login", event_type: str | None = None) -> None:
        event = self._continue_events.get(run.id)
        if not event:
            event = asyncio.Event()
            self._continue_events[run.id] = event
        event.clear()
        self._set_status(run, status)
        self._append_event(run, event_type or status, reason)
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
        if tool in {"browser.download", "python.run_safe"}:
            return True
        if tool == "windows.send_prepared_whatsapp_message":
            return True
        if tool == "desktop.press_key":
            key = str(step.get("args", {}).get("key") or "").strip().lower()
            if key == "enter" and re.search(r"\b(send|submit|post|publish|message|email|dm|text)\b", run.prompt, re.IGNORECASE):
                return True
        host = urlparse(run.current_url).netloc.lower()
        if any(host.endswith(private_host) for private_host in PRIVATE_HOSTS) and tool in {"browser.click", "browser.type", "browser.extract"}:
            return True
        return False

    async def _confirmation_payload(self, run: AutomationRun, step: dict[str, Any]) -> dict[str, Any]:
        return {"step": step, "message": self._confirmation_message(step)}

    def _confirmation_message(self, step: dict[str, Any]) -> str:
        if step["tool"] == "python.run_safe":
            return "Astra needs approval before running Python for this automation."
        if step["tool"] == "browser.download":
            return "Astra needs approval before downloading a file."
        if step["tool"] == "windows.send_prepared_whatsapp_message":
            return "Astra needs approval before sending this WhatsApp message."
        if step["tool"] == "desktop.press_key":
            return "Astra needs approval before sending or submitting this desktop action."
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
        self._sync_runtime_context_to_run(run)
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
