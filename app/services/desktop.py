import re
import os
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.models import ActionResult


@dataclass(frozen=True)
class DesktopTarget:
    aliases: tuple[str, ...]
    kind: str
    target: str
    label: str


SAFE_TARGETS = [
    DesktopTarget(("youtube", "you tube", "yt"), "url", "https://www.youtube.com/", "YouTube"),
    DesktopTarget(("google", "google search"), "url", "https://www.google.com/", "Google"),
    DesktopTarget(("gmail", "mail"), "url", "https://mail.google.com/", "Gmail"),
    DesktopTarget(("google docs", "docs"), "url", "https://docs.google.com/", "Google Docs"),
    DesktopTarget(("google drive", "drive"), "url", "https://drive.google.com/", "Google Drive"),
    DesktopTarget(("calendar", "google calendar"), "url", "https://calendar.google.com/", "Google Calendar"),
    DesktopTarget(("notepad",), "app", "notepad.exe", "Notepad"),
    DesktopTarget(("calculator", "calc"), "app", "calc.exe", "Calculator"),
    DesktopTarget(("file explorer", "explorer", "files"), "app", "explorer.exe", "File Explorer"),
]


class DesktopActionService:
    """Small allowlisted desktop control layer for safe user-initiated actions."""

    def detect(self, text: str) -> DesktopTarget | None:
        normalized = self._normalize(text)
        if not re.search(r"\b(open|launch|start|go to|visit)\b", normalized):
            return None

        direct_url = self._extract_direct_url(text)
        if direct_url:
            return DesktopTarget((direct_url,), "url", direct_url, direct_url)

        best: tuple[int, DesktopTarget] | None = None
        for target in SAFE_TARGETS:
            for alias in target.aliases:
                alias_normalized = self._normalize(alias)
                if alias_normalized and re.search(rf"\b{re.escape(alias_normalized)}\b", normalized):
                    specificity = len(re.sub(r"[^a-z0-9]+", "", alias_normalized))
                    if best is None or specificity > best[0]:
                        best = (specificity, target)
        return best[1] if best else None

    def plan_message(self, text: str) -> ActionResult | None:
        normalized = self._normalize(text)
        if not re.search(r"\b(open|launch|start|control|click|type|move|close)\b", normalized):
            return None

        return ActionResult(
            ok=False,
            action="desktop_action",
            target="not_allowlisted",
            message=(
                "I can plan this, but I did not execute it because it is outside the current safe allowlist. "
                "Allowed actions include opening YouTube, Google, Gmail, Docs, Drive, Calendar, Notepad, Calculator, and File Explorer."
            ),
        )

    async def execute(self, text: str) -> ActionResult:
        target = self.detect(text)
        if not target:
            planned = self.plan_message(text)
            if planned:
                return planned
            return ActionResult(ok=False, action="desktop_action", target="", message="No desktop action was detected.")

        return await self.open_target(target)

    async def open_target(self, target: DesktopTarget) -> ActionResult:
        if target.kind == "app" and target.target.lower() == "explorer.exe":
            return await self.open_file_explorer()

        try:
            if target.kind == "url":
                webbrowser.open(target.target, new=2)
            elif target.kind == "app":
                self._open_app(target)
            else:
                raise ValueError(f"Unsupported target kind: {target.kind}")
        except Exception as exc:
            return ActionResult(
                ok=False,
                action="desktop_action",
                target=target.label,
                message=f"I found {target.label}, but Windows could not open it: {exc}",
            )

        return ActionResult(
            ok=True,
            action="open",
            target=target.label,
            message=f"Opened {target.label}.",
        )

    async def open_file_explorer(self, drive: str | None = None) -> ActionResult:
        location = self._file_explorer_location(drive)
        label = f"{drive.upper()}: drive" if drive else "File Explorer"
        if drive and not Path(location).exists():
            return ActionResult(
                ok=False,
                action="open",
                target=label,
                message=f"I understood {drive.upper()}: drive, but that drive is not available on this PC.",
            )

        try:
            self._open_file_explorer_location(location)
        except Exception as exc:
            return ActionResult(
                ok=False,
                action="desktop_action",
                target=label,
                message=f"I found {label}, but Windows could not open it: {exc}",
            )

        return ActionResult(ok=True, action="open", target=label, message=f"Opened {label}.")

    def _extract_direct_url(self, text: str) -> str | None:
        match = re.search(r"https?://[^\s]+", text.strip(), re.IGNORECASE)
        if not match:
            return None

        candidate = match.group(0).rstrip(".,)")
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return candidate
        return None

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _open_app(self, target: DesktopTarget) -> None:
        if sys.platform.startswith("win") and target.target.lower() == "explorer.exe":
            self._open_file_explorer_location(str(Path.home()))
            return

        self._popen([target.target])

    def _file_explorer_location(self, drive: str | None = None) -> str:
        if drive:
            return f"{drive.upper()}:\\"
        return str(Path.home())

    def _open_file_explorer_location(self, location: str) -> None:
        if sys.platform.startswith("win") and hasattr(os, "startfile"):
            os.startfile(location)  # type: ignore[attr-defined]
            return
        if sys.platform.startswith("win"):
            self._popen(["explorer.exe", location])
            return
        self._popen([location])

    def _popen(self, argv: list[str]) -> subprocess.Popen:
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        return subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
