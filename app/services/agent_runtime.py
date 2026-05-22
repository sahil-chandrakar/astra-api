from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import AutomationArtifact
from app.services.artifacts import ArtifactService
from app.services.llm import LlmService


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    description: str
    risk: str = "safe_auto"
    params_schema: dict[str, Any] = field(default_factory=dict)
    examples: tuple[str, ...] = ()


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        self._tools[definition.id] = definition

    def has(self, tool_id: str) -> bool:
        return tool_id in self._tools

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[key] for key in sorted(self._tools)]

    def planner_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": tool.id,
                "description": tool.description,
                "risk": tool.risk,
                "params_schema": tool.params_schema,
                "examples": list(tool.examples),
            }
            for tool in self.definitions()
        ]


class AgentRuntime:
    """Shared tool-planning/runtime helpers for Automation and Agent Mode."""

    def __init__(self, settings: Settings, llm: LlmService, artifacts: ArtifactService):
        self.settings = settings
        self.llm = llm
        self.artifacts = artifacts
        self.registry = self._build_registry()

    async def plan(self, prompt: str, current_url: str = "", allow_llm: bool = True) -> list[dict[str, Any]]:
        local_plan = self.local_plan(prompt)
        if local_plan:
            return local_plan
        if allow_llm and self._llm_available():
            steps = await self._plan_with_llm(prompt, current_url)
            if steps:
                return steps
        return []

    def local_plan(self, prompt: str) -> list[dict[str, Any]]:
        return self._local_generic_plan(prompt)

    async def plan_next_action(self, prompt: str, agent_state: dict[str, Any], current_url: str = "", allow_llm: bool = True) -> dict[str, Any] | None:
        local_action = self._local_next_action(prompt, agent_state)
        if local_action:
            return local_action
        if allow_llm and self._llm_available():
            return await self._plan_next_with_llm(prompt, agent_state, current_url)
        return None

    def tool_ids(self) -> set[str]:
        return {tool.id for tool in self.registry.definitions()}

    def recent_artifact_context(self, limit: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "id": artifact.id,
                "kind": artifact.kind,
                "media_type": artifact.media_type,
                "title": artifact.title,
                "filename": artifact.filename,
                "source_url": artifact.source_url,
                "created_at": artifact.created_at.isoformat(),
            }
            for artifact in self.artifacts.list_recent(limit=limit)
        ]

    def resolve_artifact(self, query: str, media_types: list[str] | None = None) -> tuple[AutomationArtifact | None, list[AutomationArtifact], str]:
        return self.artifacts.resolve_reference(query, media_types=media_types)

    def resolve_app(self, app_name: str) -> dict[str, Any]:
        clean = re.sub(r"[^a-zA-Z0-9._ -]+", " ", app_name or "").strip()
        if not clean:
            clean = "default"
        if clean.lower() in {"default", "system", "associated app"}:
            return {"app_name": "default", "path": "", "launch_mode": "default"}
        if sys.platform.startswith("win") and clean.lower() in {"whatsapp", "whats app", "whatsapp desktop"}:
            return {"app_name": "WhatsApp", "path": "whatsapp://", "launch_mode": "uri"}

        executable_names = self._candidate_executables(clean)
        for name in executable_names:
            if path := shutil.which(name):
                return {"app_name": clean, "path": path, "launch_mode": "executable"}
        if sys.platform.startswith("win"):
            if path := self._resolve_windows_app_path(executable_names):
                return {"app_name": clean, "path": path, "launch_mode": "executable"}
        return {"app_name": clean, "path": "", "launch_mode": "missing"}

    def open_artifact_with_app(self, artifact: AutomationArtifact, app_name: str = "default") -> dict[str, Any]:
        target = self.artifacts.ensure_artifact_path(artifact)
        app = self.resolve_app(app_name)
        launch_mode = str(app.get("launch_mode") or "")
        if launch_mode == "missing":
            raise ValueError(f"Astra resolved {artifact.filename}, but could not find {app.get('app_name') or app_name} on this PC.")

        if launch_mode == "default":
            self._open_default(target)
        else:
            self._popen([str(app["path"]), str(target)])
        return {
            "artifact_id": artifact.id,
            "filename": artifact.filename,
            "path": str(target),
            "app": app,
        }

    def open_artifact_containing_folder(self, artifact: AutomationArtifact) -> dict[str, Any]:
        target = self.artifacts.ensure_artifact_path(artifact)
        folder = target.parent
        if sys.platform.startswith("win"):
            self._popen(["explorer.exe", f"/select,{target}"])
        elif sys.platform == "darwin":
            self._popen(["open", "-R", str(target)])
        else:
            self._popen(["xdg-open", str(folder)])
        return {
            "artifact_id": artifact.id,
            "filename": artifact.filename,
            "path": str(target),
            "folder_path": str(folder),
        }

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        register = registry.register
        register(ToolDefinition("artifact.resolve_reference", "Resolve references like 'that download', 'latest video', or title fragments to an Astra-owned artifact.", params_schema={"query": "string", "media_types": "array"}))
        register(ToolDefinition("artifact.list_recent", "List recent Astra-owned artifacts with optional media filters.", params_schema={"media_types": "array", "limit": "integer"}))
        register(ToolDefinition("artifact.pick", "Pick one artifact from user-selected candidates.", params_schema={"artifact_id": "string"}))
        register(ToolDefinition("artifact.open_containing_folder", "Reveal an Astra-owned artifact in its containing folder.", params_schema={"artifact_id": "string", "reference": "string"}))
        register(ToolDefinition("file.search_scoped", "Search only Astra-owned artifact/download folders for a file.", params_schema={"query": "string", "media_types": "array"}))
        register(ToolDefinition("file.open", "Open an Astra-owned artifact with the default system app.", params_schema={"artifact_id": "string", "reference": "string"}))
        register(ToolDefinition("app.resolve", "Resolve an app name such as VLC or default into a local launcher.", params_schema={"app_name": "string"}))
        register(ToolDefinition("app.open", "Open a local desktop app by name using an allowlisted launcher.", params_schema={"app_name": "string"}))
        register(ToolDefinition("app.open_with_file", "Open an Astra-owned artifact/file with a resolved app.", params_schema={"artifact_id": "string", "reference": "string", "app_name": "string"}))
        register(ToolDefinition("desktop.find_text", "Find visible desktop text/control text in the active app.", params_schema={"text": "string", "app_name": "string", "control_types": "array", "exclude_control_types": "array"}))
        register(ToolDefinition("desktop.click", "Click a previously found desktop target or visible text.", params_schema={"target": "object|string", "text": "string", "button": "left|right"}))
        register(ToolDefinition("desktop.type_text", "Type text into the focused desktop control.", params_schema={"text": "string"}))
        register(ToolDefinition("desktop.press_key", "Press a safe desktop key such as Enter, Tab, or Escape.", risk="safe_confirm", params_schema={"key": "string"}))
        register(ToolDefinition("desktop.verify_text", "Verify expected text is visible in the desktop UI.", params_schema={"text": "string", "app_name": "string"}))
        register(ToolDefinition("browser.open", "Open a URL in the automation browser.", params_schema={"url": "string"}))
        register(ToolDefinition("browser.search", "Search Google or YouTube.", params_schema={"site": "google|youtube", "query": "string"}))
        register(ToolDefinition("browser.extract", "Extract visible browser page content.", params_schema={"target": "string"}))
        register(ToolDefinition("video.download", "Download a verified public video URL into Astra's artifact store.", params_schema={"url": "string"}))
        register(ToolDefinition("video.download_permitted", "Compatibility alias for downloading a verified public video URL into Astra's artifact store.", params_schema={"url": "string"}))
        register(ToolDefinition("python.run_safe", "Run constrained Python automation code after approval.", risk="safe_confirm", params_schema={"code": "string"}))
        register(ToolDefinition("system.ask_user", "Ask the user for a missing value only when context and artifacts cannot resolve it.", params_schema={"message": "string"}))
        register(ToolDefinition("task.finish", "Finish the current task after verification succeeds.", params_schema={"message": "string"}))
        register(ToolDefinition("task.replan", "Discard the current failed action and plan the next best action.", params_schema={"reason": "string"}))
        return registry

    def _local_generic_plan(self, prompt: str) -> list[dict[str, Any]]:
        normalized = prompt.lower()
        wants_open = bool(re.search(r"\b(open|play|view|watch|listen|launch)\b", normalized))
        artifact_ref = bool(re.search(r"\b(that|latest|recent|previous|last|downloaded|download|artifact|file|song|video|audio|music)\b", normalized))
        if wants_open and artifact_ref and not re.search(r"https?://", prompt):
            app_name = self._extract_app_name(prompt)
            media_types = self.artifacts.infer_media_types(prompt)
            final_tool = "artifact.open_containing_folder" if self._wants_containing_folder(prompt) else "app.open_with_file"
            return [
                {
                    "tool": "artifact.resolve_reference",
                    "description": "Resolve the referenced Astra artifact.",
                    "args": {"query": prompt, "media_types": media_types},
                },
                {
                    "tool": final_tool,
                    "description": "Open the folder containing the resolved artifact." if final_tool == "artifact.open_containing_folder" else f"Open the resolved artifact with {app_name if app_name != 'default' else 'the default app'}.",
                    "args": {} if final_tool == "artifact.open_containing_folder" else {"app_name": app_name},
                },
            ]
        return []

    def _local_next_action(self, prompt: str, agent_state: dict[str, Any]) -> dict[str, Any] | None:
        last_tool = str(agent_state.get("last_tool") or "")
        last_status = str(agent_state.get("last_tool_status") or "")
        if last_status == "success" and last_tool in {"app.open_with_file", "file.open", "artifact.open_containing_folder"}:
            return {
                "tool": "task.finish",
                "description": "Finish after verifying the requested local action.",
                "args": {"message": str(agent_state.get("last_result") or "Automation completed.")},
            }

        selected_artifact_id = str(agent_state.get("selected_artifact_id") or "").strip()
        if selected_artifact_id:
            return {
                "tool": "artifact.pick",
                "description": "Use the artifact selected by the user.",
                "args": {"artifact_id": selected_artifact_id},
            }

        normalized = prompt.lower()
        wants_open = bool(re.search(r"\b(open|play|view|watch|listen|launch|show|reveal|find)\b", normalized))
        artifact_ref = bool(re.search(r"\b(that|latest|recent|previous|last|downloaded|download|artifact|file|song|video|audio|music|folder|directory|located|location)\b", normalized))
        if wants_open and artifact_ref and not re.search(r"https?://", prompt):
            if not str(agent_state.get("last_artifact_id") or ""):
                query = str(agent_state.get("pending_input") or prompt).strip()
                if self._is_continue_ack(query):
                    query = prompt
                return {
                    "tool": "artifact.resolve_reference",
                    "description": "Resolve the referenced Astra artifact.",
                    "args": {"query": query, "media_types": self.artifacts.infer_media_types(prompt)},
                }
            if self._wants_containing_folder(prompt):
                return {
                    "tool": "artifact.open_containing_folder",
                    "description": "Open the folder containing the resolved artifact.",
                    "args": {"artifact_id": str(agent_state.get("last_artifact_id") or "")},
                }
            return {
                "tool": "app.open_with_file",
                "description": "Open the resolved artifact with the requested app.",
                "args": {"artifact_id": str(agent_state.get("last_artifact_id") or ""), "app_name": self._extract_app_name(prompt)},
            }

        return None

    async def _plan_with_llm(self, prompt: str, current_url: str) -> list[dict[str, Any]]:
        system_prompt = (
            "You are Astra's universal tool planner. Return strict JSON only with a top-level steps array. "
            "Use only registered tools. Prefer generic artifact/app tools over phrase-specific behavior. "
            "Resolve words like 'that', 'latest', 'previous', and 'downloaded' with artifact.resolve_reference before opening files. "
            "Use system.ask_user only when recent artifacts and prompt context cannot resolve required inputs. "
            "Never invent local paths or shell commands."
        )
        user_prompt = json.dumps(
            {
                "prompt": prompt,
                "current_url": current_url,
                "recent_artifacts": self.recent_artifact_context(),
                "registered_tools": self.registry.planner_payload(),
                "output_schema": {"steps": [{"tool": "registered tool id", "description": "short text", "args": "object"}]},
            },
            ensure_ascii=True,
        )
        raw, setup = await self.llm.complete(system_prompt, user_prompt, model=self._planner_model())
        if setup:
            return []
        payload = self._extract_json(raw)
        steps = payload.get("steps") if isinstance(payload, dict) else None
        return steps if isinstance(steps, list) else []

    async def _plan_next_with_llm(self, prompt: str, agent_state: dict[str, Any], current_url: str) -> dict[str, Any] | None:
        system_prompt = (
            "You are Astra's observe-plan-act planner. Return strict JSON for exactly one next_action. "
            "Use only registered tools. Prefer generic artifact/app tools. "
            "If a step just succeeded and the user goal is satisfied, use task.finish. "
            "If context is missing, use system.ask_user. Never invent local paths or shell commands."
        )
        user_prompt = json.dumps(
            {
                "prompt": prompt,
                "current_url": current_url,
                "agent_state": agent_state,
                "recent_artifacts": self.recent_artifact_context(),
                "registered_tools": self.registry.planner_payload(),
                "output_schema": {"next_action": {"tool": "registered tool id", "description": "short text", "args": "object"}},
            },
            ensure_ascii=True,
        )
        raw, setup = await self.llm.complete(system_prompt, user_prompt, model=self._planner_model())
        if setup:
            return None
        payload = self._extract_json(raw)
        action = payload.get("next_action") if isinstance(payload, dict) else None
        return action if isinstance(action, dict) else None

    def _llm_available(self) -> bool:
        model = self._planner_model()
        provider = model.split(":", 1)[0].strip().lower() if ":" in model else "cerebras"
        configured = getattr(self.llm, "provider_configured", None)
        return bool(configured(provider)) if callable(configured) else self.llm.model_configured(model)

    def _planner_model(self) -> str:
        selector = getattr(self.llm, "model_for_profile", None)
        return selector("pro") if callable(selector) else self.settings.resolved_cerebras_pro_model

    def _extract_app_name(self, prompt: str) -> str:
        normalized = prompt.lower()
        match = re.search(r"\b(?:in|with|using)\s+([a-zA-Z0-9 ._-]{2,40})\b", prompt, re.IGNORECASE)
        if match:
            candidate = re.sub(r"\b(app|player|please)\b", " ", match.group(1), flags=re.IGNORECASE)
            candidate = re.sub(r"\s+", " ", candidate).strip(" .")
            if candidate:
                return candidate
        if "vlc" in normalized:
            return "vlc"
        return "default"

    def _wants_containing_folder(self, prompt: str) -> bool:
        normalized = prompt.lower()
        return bool(re.search(r"\b(folder|directory|where|located|location|containing|reveal|show in folder)\b", normalized))

    def _is_continue_ack(self, text: str) -> bool:
        compact = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        return compact in {"", "continue", "ready", "user is ready to continue", "ok", "okay", "yes"}

    def _candidate_executables(self, app_name: str) -> list[str]:
        clean = app_name.strip().lower()
        names = [clean]
        if not clean.endswith(".exe"):
            names.append(f"{clean}.exe")
        if clean == "vlc":
            names.extend(["vlc.exe", "vlc"])
        return list(dict.fromkeys(names))

    def _resolve_windows_app_path(self, executable_names: list[str]) -> str | None:
        common_roots = [
            Path(os.environ.get("ProgramFiles", "C:\\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")),
            Path(os.environ.get("LOCALAPPDATA", "")),
        ]
        common_patterns = [
            ("vlc.exe", ("VideoLAN", "VLC", "vlc.exe")),
        ]
        for executable in executable_names:
            for expected, parts in common_patterns:
                if executable.lower() != expected:
                    continue
                for root in common_roots:
                    candidate = root.joinpath(*parts)
                    if candidate.exists():
                        return str(candidate)
            if path := self._resolve_windows_app_path_from_registry(executable):
                return path
        return None

    def _resolve_windows_app_path_from_registry(self, executable: str) -> str | None:
        try:
            import winreg

            subkey = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable}"
            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(root, subkey) as key:
                        value, _kind = winreg.QueryValueEx(key, "")
                        if value and Path(value).exists():
                            return str(value)
                except OSError:
                    continue
        except Exception:
            return None
        return None

    def _open_default(self, target: Path) -> None:
        if sys.platform.startswith("win") and hasattr(os, "startfile"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            self._popen(["open", str(target)])
        else:
            self._popen(["xdg-open", str(target)])

    def _popen(self, argv: list[str]) -> subprocess.Popen:
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creation_flags)

    def _extract_json(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            payload = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return {}
            try:
                payload = json.loads(match.group(0))
            except Exception:
                return {}
        return payload if isinstance(payload, dict) else {}
