import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from app.config import Settings
from app.models import AutomationEngineStatus, AutomationRecipe


OPENRPA_INPUT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OpenRPAError(RuntimeError):
    """Base error for OpenRPA execution failures."""


class OpenRPAMissingError(OpenRPAError):
    """Raised when OpenRPA.exe cannot be found."""


class OpenRPACancelledError(OpenRPAError):
    """Raised when a running OpenRPA workflow is cancelled."""


@dataclass
class OpenRPARunResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    command: list[str]


OpenRPAEventCallback = Callable[[str, str, dict[str, Any]], None]


class OpenRPAAdapter:
    def __init__(self, settings: Settings):
        self.settings = settings

    def status(self) -> AutomationEngineStatus:
        configured_path = self._configured_path()
        if configured_path:
            candidate = Path(configured_path).expanduser()
            if candidate.is_file():
                return AutomationEngineStatus(
                    id="openrpa",
                    label="OpenRPA",
                    installed=True,
                    configured_path=str(candidate),
                    message="OpenRPA executable is configured.",
                )
            return AutomationEngineStatus(
                id="openrpa",
                label="OpenRPA",
                installed=False,
                configured_path=str(candidate),
                message="ASTRA_OPENRPA_EXE points to a file that does not exist.",
            )

        discovered = self._discover_executable()
        if discovered:
            return AutomationEngineStatus(
                id="openrpa",
                label="OpenRPA",
                installed=True,
                configured_path=discovered,
                message="OpenRPA executable was discovered automatically.",
            )

        return AutomationEngineStatus(
            id="openrpa",
            label="OpenRPA",
            installed=False,
            configured_path="",
            message="OpenRPA.exe was not found. Install OpenRPA or set ASTRA_OPENRPA_EXE to the executable path.",
        )

    def build_command(self, recipe: AutomationRecipe, inputs: dict[str, Any] | None = None) -> list[str]:
        exe = self.executable_path()
        self.validate_recipe(recipe)
        command = [exe, "/WorkflowID", recipe.workflow_ref.strip()]
        allowed_inputs = set(recipe.inputs)
        safe_inputs = inputs or {}
        for key in recipe.inputs:
            if key not in safe_inputs:
                continue
            command.extend([f"-{key}", self._stringify_input(safe_inputs[key])])
        unexpected = sorted(str(key) for key in safe_inputs if str(key) not in allowed_inputs)
        if unexpected:
            raise ValueError(f"OpenRPA input is not registered for this workflow: {', '.join(unexpected[:5])}.")
        return command

    def run(
        self,
        recipe: AutomationRecipe,
        inputs: dict[str, Any] | None,
        cancel_event: threading.Event,
        on_event: OpenRPAEventCallback | None = None,
    ) -> OpenRPARunResult:
        command = self.build_command(recipe, inputs)
        timeout_seconds = max(1, int(recipe.timeout_seconds or 300))
        started = time.monotonic()
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        def read_stream(stream: Any, sink: list[str], event_type: str) -> None:
            if stream is None:
                return
            try:
                for line in iter(stream.readline, ""):
                    clean = line.rstrip()
                    if not clean:
                        continue
                    sink.append(clean)
                    if on_event:
                        on_event(event_type, clean[-1000:], {"stream": event_type})
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        try:
            while True:
                if cancel_event.is_set():
                    self._kill_process(process)
                    raise OpenRPACancelledError("OpenRPA workflow cancelled by user.")
                returncode = process.poll()
                if returncode is not None:
                    break
                if time.monotonic() - started > timeout_seconds:
                    self._kill_process(process)
                    raise TimeoutError(f"OpenRPA workflow exceeded the {timeout_seconds} second timeout.")
                time.sleep(0.12)
        finally:
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)

        return OpenRPARunResult(
            returncode=int(process.returncode or 0),
            stdout="\n".join(stdout_lines).strip(),
            stderr="\n".join(stderr_lines).strip(),
            elapsed_seconds=time.monotonic() - started,
            command=self._redacted_command(command, recipe.inputs),
        )

    def executable_path(self) -> str:
        status = self.status()
        if not status.installed or not status.configured_path:
            raise OpenRPAMissingError(status.message)
        return status.configured_path

    def validate_recipe(self, recipe: AutomationRecipe) -> None:
        if recipe.engine != "openrpa":
            raise ValueError("OpenRPA adapter can only run recipes with engine='openrpa'.")
        workflow_ref = recipe.workflow_ref.strip()
        if not workflow_ref:
            raise ValueError("OpenRPA workflow_ref is required.")
        if recipe.workflow_ref_type == "filename":
            self._validate_relative_xaml(workflow_ref)
        elif recipe.workflow_ref_type == "id":
            self._validate_workflow_id(workflow_ref)
        else:
            raise ValueError("OpenRPA workflow_ref_type must be 'id' or 'filename'.")
        for input_name in recipe.inputs:
            if not OPENRPA_INPUT_NAME_PATTERN.fullmatch(input_name):
                raise ValueError(f"OpenRPA input name is invalid: {input_name}.")

    def _configured_path(self) -> str:
        return (getattr(self.settings, "astra_openrpa_exe", "") or os.getenv("ASTRA_OPENRPA_EXE") or "").strip().strip('"')

    def _discover_executable(self) -> str:
        for name in ("OpenRPA.exe", "OpenRPA"):
            if found := shutil.which(name):
                return found

        candidates: list[Path] = []
        for env_name in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)"):
            root = os.getenv(env_name)
            if not root:
                continue
            candidates.extend(
                [
                    Path(root) / "Programs" / "OpenRPA" / "OpenRPA.exe",
                    Path(root) / "OpenRPA" / "OpenRPA.exe",
                    Path(root) / "openrpa" / "OpenRPA.exe",
                ]
            )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return ""

    def _validate_relative_xaml(self, workflow_ref: str) -> None:
        if re.match(r"^[A-Za-z]:[\\/]", workflow_ref) or workflow_ref.startswith(("/", "\\")):
            raise ValueError("OpenRPA workflow filenames must be relative catalog references.")
        normalized = workflow_ref.replace("\\", "/")
        windows_parts = PureWindowsPath(workflow_ref).parts
        posix_parts = PurePosixPath(normalized).parts
        if ".." in windows_parts or ".." in posix_parts:
            raise ValueError("OpenRPA workflow filenames cannot contain parent directory segments.")
        if not workflow_ref.lower().endswith(".xaml"):
            raise ValueError("OpenRPA workflow filename references must end with .xaml.")

    def _validate_workflow_id(self, workflow_ref: str) -> None:
        if re.match(r"^[A-Za-z]:[\\/]", workflow_ref) or "/" in workflow_ref or "\\" in workflow_ref:
            raise ValueError("OpenRPA workflow IDs cannot be file paths.")
        if workflow_ref.lower().endswith(".xaml"):
            raise ValueError("Use workflow_ref_type='filename' for .xaml workflow references.")

    def _stringify_input(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)[:2000]

    def _kill_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            process.kill()
        except Exception:
            return
        try:
            process.wait(timeout=2)
        except Exception:
            pass

    def _redacted_command(self, command: list[str], input_names: list[str]) -> list[str]:
        redacted = list(command)
        sensitive_names = {name.lower() for name in input_names if re.search(r"(password|token|secret|credential)", name, re.IGNORECASE)}
        for index, item in enumerate(redacted[:-1]):
            if item.startswith("-") and item[1:].lower() in sensitive_names:
                redacted[index + 1] = "[redacted]"
        return redacted
