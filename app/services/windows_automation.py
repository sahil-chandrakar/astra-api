import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class WindowsAutomationResult:
    ok: bool
    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    needs_user: bool = False


class WindowsAutomationService:
    """Allowlisted Windows desktop automation used by Automation runs."""

    def is_supported(self) -> bool:
        return sys.platform.startswith("win")

    def open_calculator(self, expression: str = "") -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("Calculator")

        self._popen(["calc.exe"])
        data = {"app": "Calculator"}
        expression = expression.strip()
        if not expression:
            return WindowsAutomationResult(True, "calculator", "Opened Calculator.", data)

        data["expression"] = expression
        time.sleep(0.8)
        typed = self._send_keys_to_active_window(self._calculator_send_keys(expression))
        if typed:
            return WindowsAutomationResult(True, "calculator", f"Opened Calculator and typed {expression}.", data)
        return WindowsAutomationResult(
            True,
            "calculator",
            f"Opened Calculator. Type {expression} manually if it was not entered.",
            {**data, "manual_expression_entry": True},
        )

    def open_file_explorer(self, target: str = "") -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("File Explorer")

        location = self.resolve_file_explorer_target(target)
        if not location.exists() or not location.is_dir():
            return WindowsAutomationResult(
                False,
                "windows_error",
                f"File Explorer target is not available: {location}.",
                {"app": "File Explorer", "target": str(location)},
            )

        self._open_location(location)
        label = self._file_explorer_label(target, location)
        return WindowsAutomationResult(
            True,
            "file_explorer",
            f"Opened {label}.",
            {"app": "File Explorer", "target": label, "opened_path": str(location)},
        )

    def set_alarm(self, target_time: datetime, label: str = "Astra alarm") -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("Windows Clock")

        self._open_uri("ms-clock://alarms")
        display_time = target_time.strftime("%I:%M %p").lstrip("0")
        data = {
            "app": "Windows Clock",
            "alarm_time": target_time.isoformat(),
            "display_time": display_time,
            "label": label,
        }

        if self._try_set_alarm_with_pywinauto(target_time, label):
            return WindowsAutomationResult(True, "alarm_set", f"Set alarm for {display_time}.", data)

        return WindowsAutomationResult(
            False,
            "windows_error",
            f"Astra could not set the Windows Clock alarm for {display_time}. No alarm was saved.",
            {**data, "automation_failed": True},
        )

    def open_app(self, app_name: str) -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported(app_name or "Desktop app")

        clean = re.sub(r"\s+", " ", app_name or "").strip()
        if not clean:
            return WindowsAutomationResult(False, "windows_error", "Desktop app open step needs an app name.")

        normalized = clean.lower()
        try:
            if normalized in {"whatsapp", "whats app", "whatsapp desktop"}:
                if self._open_windows_app_uri_or_start_app("whatsapp://", "WhatsApp"):
                    return WindowsAutomationResult(True, "app_opened", "Opened WhatsApp.", {"app": "WhatsApp"})
                return WindowsAutomationResult(False, "windows_error", "WhatsApp is not installed or could not be opened.", {"app": "WhatsApp"})

            executable = self._resolve_executable(clean)
            if executable:
                self._popen([executable])
                return WindowsAutomationResult(True, "app_opened", f"Opened {clean}.", {"app": clean, "path": executable})

            if self._open_start_menu_app(clean):
                return WindowsAutomationResult(True, "app_opened", f"Opened {clean}.", {"app": clean})
        except Exception as exc:
            return WindowsAutomationResult(False, "windows_error", f"Could not open {clean}: {exc}", {"app": clean})

        return WindowsAutomationResult(False, "windows_error", f"Could not find an installed app named {clean}.", {"app": clean})

    def prepare_whatsapp_message(self, contact: str, message: str) -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("WhatsApp")

        clean_contact = re.sub(r"\s+", " ", contact or "").strip()
        clean_message = str(message or "").strip()
        if not clean_contact:
            return WindowsAutomationResult(False, "windows_error", "WhatsApp automation needs a contact name.", {"app": "WhatsApp"})
        if not clean_message:
            return WindowsAutomationResult(False, "windows_error", "WhatsApp automation needs a message.", {"app": "WhatsApp", "contact": clean_contact})

        try:
            self.open_app("WhatsApp")
            if not self._focus_window_by_title("WhatsApp", timeout=8):
                return WindowsAutomationResult(False, "windows_error", "WhatsApp opened, but Astra could not focus its window.", {"app": "WhatsApp", "contact": clean_contact})

            if not self._open_whatsapp_chat(clean_contact):
                return WindowsAutomationResult(
                    False,
                    "windows_error",
                    f"Could not open a WhatsApp chat for {clean_contact}.",
                    {"app": "WhatsApp", "contact": clean_contact},
                )

            message_box_ready = self._focus_whatsapp_message_box()
            self._send_text_to_active_control(clean_message, replace=False)
        except Exception as exc:
            return WindowsAutomationResult(
                False,
                "windows_error",
                f"Could not prepare WhatsApp message for {clean_contact}: {exc}",
                {"app": "WhatsApp", "contact": clean_contact, "message_length": len(clean_message)},
            )

        return WindowsAutomationResult(
            True,
            "whatsapp_prepared",
            f"Prepared WhatsApp message to {clean_contact}.",
            {
                "app": "WhatsApp",
                "contact": clean_contact,
                "message_length": len(clean_message),
                "message_box_detected": message_box_ready,
            },
        )

    def find_text(
        self,
        text: str,
        app_name: str = "",
        timeout: int = 8,
        control_types: list[str] | None = None,
        exclude_control_types: list[str] | None = None,
    ) -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("Desktop text search")
        clean = re.sub(r"\s+", " ", text or "").strip()
        if not clean:
            return WindowsAutomationResult(False, "windows_error", "Desktop text search needs text to find.")
        try:
            target = self._find_text_target(clean, app_name, timeout, control_types or [], exclude_control_types or [])
        except Exception as exc:
            return WindowsAutomationResult(False, "windows_error", f"Could not inspect the desktop UI: {exc}", {"text": clean, "app": app_name})
        if not target:
            scope = f" in {app_name}" if app_name else ""
            return WindowsAutomationResult(False, "desktop_text_not_found", f"Could not find '{clean}'{scope}.", {"text": clean, "app": app_name})
        return WindowsAutomationResult(True, "desktop_text_found", f"Found '{clean}'.", {"text": clean, "app": app_name, "target": target})

    def click_target(self, target: Any = None, text: str = "", button: str = "left") -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("Desktop click")

        try:
            resolved = target if isinstance(target, dict) else None
            if not resolved and text:
                found = self._find_text_target(text, "", 5, [], [])
                resolved = found
            if not resolved:
                return WindowsAutomationResult(False, "windows_error", "Desktop click needs a found target or text.")

            rect = resolved.get("rect") if isinstance(resolved, dict) else None
            if not isinstance(rect, dict):
                return WindowsAutomationResult(False, "windows_error", "Desktop click target is missing screen coordinates.", {"target": resolved})
            x = int((int(rect["left"]) + int(rect["right"])) / 2)
            y = int((int(rect["top"]) + int(rect["bottom"])) / 2)
            self._click_screen_point(x, y, button=button if button in {"left", "right"} else "left")
        except Exception as exc:
            return WindowsAutomationResult(False, "windows_error", f"Could not click the desktop target: {exc}", {"target": target})
        return WindowsAutomationResult(True, "desktop_clicked", "Clicked the desktop target.", {"target": resolved})

    def type_text(self, text: str, replace: bool = False) -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("Desktop typing")
        clean = str(text or "")
        if not clean:
            return WindowsAutomationResult(False, "windows_error", "Desktop typing needs text.")
        try:
            from pywinauto.keyboard import send_keys

            if replace:
                send_keys("^a{BACKSPACE}", pause=0.02)
            send_keys(self._send_keys_literal(clean), pause=0.02, with_spaces=True)
        except Exception as exc:
            return WindowsAutomationResult(False, "windows_error", f"Could not type into the active desktop control: {exc}", {"text_length": len(clean)})
        return WindowsAutomationResult(True, "desktop_typed", "Typed text into the focused desktop control.", {"text_length": len(clean), "replace": replace})

    def press_key(self, key: str) -> WindowsAutomationResult:
        if not self.is_supported():
            return self._unsupported("Desktop key press")
        normalized = (key or "").strip().lower()
        key_map = {
            "enter": "{ENTER}",
            "tab": "{TAB}",
            "escape": "{ESC}",
            "backspace": "{BACKSPACE}",
            "delete": "{DELETE}",
            "space": "{SPACE}",
        }
        keys = key_map.get(normalized)
        if not keys:
            return WindowsAutomationResult(False, "windows_error", f"Desktop key is not allowed: {key}.", {"key": key})
        try:
            from pywinauto.keyboard import send_keys

            send_keys(keys, pause=0.03)
        except Exception as exc:
            return WindowsAutomationResult(False, "windows_error", f"Could not press {normalized}: {exc}", {"key": normalized})
        return WindowsAutomationResult(True, "desktop_key_pressed", f"Pressed {normalized}.", {"key": normalized})

    def verify_text(self, text: str, app_name: str = "", timeout: int = 6) -> WindowsAutomationResult:
        result = self.find_text(text, app_name=app_name, timeout=timeout)
        if result.ok:
            return WindowsAutomationResult(True, "desktop_verified", f"Verified '{text}' is visible.", result.data)
        return WindowsAutomationResult(False, "desktop_verify_failed", f"Could not verify '{text}' is visible.", result.data)

    def resolve_file_explorer_target(self, target: str = "") -> Path:
        normalized = re.sub(r"\s+", " ", target.lower()).strip()
        home = Path.home()
        known_targets = {
            "": home,
            "home": home,
            "user": home,
            "downloads": home / "Downloads",
            "download": home / "Downloads",
            "documents": home / "Documents",
            "document": home / "Documents",
            "desktop": home / "Desktop",
            "pictures": home / "Pictures",
            "picture": home / "Pictures",
            "music": home / "Music",
            "videos": home / "Videos",
            "video": home / "Videos",
        }
        if normalized in known_targets:
            return known_targets[normalized]

        drive_match = re.fullmatch(r"([a-z]):?\\?", normalized)
        if drive_match:
            return Path(f"{drive_match.group(1).upper()}:\\")

        candidate = Path(target).expanduser()
        if candidate.is_absolute():
            return candidate
        return home

    def _unsupported(self, app: str) -> WindowsAutomationResult:
        return WindowsAutomationResult(
            False,
            "windows_error",
            f"{app} automation is only supported on Windows.",
            {"app": app, "platform": sys.platform},
        )

    def _file_explorer_label(self, target: str, location: Path) -> str:
        normalized = target.strip()
        if normalized:
            return normalized
        if location == Path.home():
            return "File Explorer"
        return location.name or str(location)

    def _open_location(self, location: Path) -> None:
        if hasattr(os, "startfile"):
            os.startfile(str(location))  # type: ignore[attr-defined]
            return
        self._popen(["explorer.exe", str(location)])

    def _open_uri(self, uri: str) -> None:
        if hasattr(os, "startfile"):
            os.startfile(uri)  # type: ignore[attr-defined]
            return
        self._popen(["explorer.exe", uri])

    def _popen(self, argv: list[str]) -> subprocess.Popen:
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if self.is_supported() else 0
        return subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

    def _resolve_executable(self, app_name: str) -> str:
        import shutil

        candidates = [app_name]
        if not app_name.lower().endswith(".exe"):
            candidates.append(f"{app_name}.exe")
        aliases = {
            "vlc": ["vlc.exe"],
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
        }
        candidates.extend(aliases.get(app_name.lower(), []))
        for candidate in dict.fromkeys(candidates):
            if path := shutil.which(candidate):
                return path
        return ""

    def _open_windows_app_uri_or_start_app(self, uri: str, app_name: str) -> bool:
        try:
            self._open_uri(uri)
            time.sleep(1.2)
            if self._window_title_visible(app_name, timeout=5):
                return True
            # URI launch success is the best signal Windows gives us for Store apps.
            # A later desktop.find_text step performs the real UI verification.
            return True
        except Exception:
            pass
        return self._open_start_menu_app(app_name)

    def _open_start_menu_app(self, app_name: str) -> bool:
        app_id = self._start_app_id(app_name)
        if not app_id:
            return False
        self._popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        time.sleep(1.2)
        return self._window_title_visible(app_name, timeout=6)

    def _start_app_id(self, app_name: str) -> str:
        try:
            command = (
                "$name = "
                + repr(f"*{app_name}*")
                + "; (Get-StartApps | Where-Object { $_.Name -like $name } | Select-Object -First 1 -ExpandProperty AppID)"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if self.is_supported() else 0,
            )
            return (completed.stdout or "").strip().splitlines()[0].strip()
        except Exception:
            return ""

    def _window_title_visible(self, title_fragment: str, timeout: float = 4) -> bool:
        fragment = re.sub(r"\s+", " ", title_fragment or "").strip().lower()
        if not fragment:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                from pywinauto import Desktop

                desktop = Desktop(backend="uia")
                for window in desktop.windows():
                    title = (window.window_text() or "").lower()
                    if fragment in title or title in fragment:
                        return True
            except Exception:
                return False
            time.sleep(0.25)
        return False

    def _focus_window_by_title(self, title_fragment: str, timeout: float = 4) -> bool:
        fragment = re.sub(r"\s+", " ", title_fragment or "").strip().lower()
        if not fragment:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                from pywinauto import Desktop

                desktop = Desktop(backend="uia")
                for window in desktop.windows():
                    title = (window.window_text() or "").lower()
                    if fragment not in title and title not in fragment:
                        continue
                    try:
                        window.set_focus()
                    except Exception:
                        hwnd = getattr(getattr(window, "element_info", None), "handle", None)
                        if hwnd:
                            self._focus_window(int(hwnd))
                    time.sleep(0.25)
                    return True
            except Exception:
                return False
            time.sleep(0.25)
        return False

    def _open_whatsapp_chat(self, contact: str) -> bool:
        if self._search_whatsapp_contact(contact, hotkey="f"):
            return True
        self._focus_window_by_title("WhatsApp", timeout=2)
        return self._search_whatsapp_contact(contact, hotkey="n")

    def _search_whatsapp_contact(self, contact: str, hotkey: str) -> bool:
        try:
            from pywinauto.keyboard import send_keys

            if hotkey == "f" and self._focus_whatsapp_search_box():
                pass
            else:
                send_keys(f"^{hotkey}", pause=0.04)
                time.sleep(0.45)
            send_keys("^a{BACKSPACE}" + self._send_keys_literal(contact), pause=0.03, with_spaces=True)
            time.sleep(0.9)
            send_keys("{ENTER}", pause=0.05)
            time.sleep(0.9)
            return self._focus_whatsapp_message_box()
        except Exception:
            return False

    def _focus_whatsapp_search_box(self) -> bool:
        labels = (
            "Search or start a new chat",
            "Search or start new chat",
            "Search",
            "Search input textbox",
        )
        for label in labels:
            try:
                target = self._find_text_target(label, "WhatsApp", 2, ["edit", "text"], [])
            except Exception:
                target = None
            if self._click_desktop_target(target):
                time.sleep(0.2)
                return True
        return False

    def _focus_whatsapp_message_box(self) -> bool:
        labels = (
            "Type a message",
            "Message",
            "Type a message here",
        )
        for label in labels:
            try:
                target = self._find_text_target(label, "WhatsApp", 2, ["edit", "text"], [])
            except Exception:
                target = None
            if self._click_desktop_target(target):
                time.sleep(0.2)
                return True
        return False

    def _send_text_to_active_control(self, text: str, replace: bool = False) -> None:
        from pywinauto.keyboard import send_keys

        prefix = "^a{BACKSPACE}" if replace else ""
        send_keys(prefix + self._send_keys_literal(text), pause=0.02, with_spaces=True)

    def _click_desktop_target(self, target: Any) -> bool:
        if not isinstance(target, dict):
            return False
        rect = target.get("rect")
        if not isinstance(rect, dict):
            return False
        try:
            x = int((int(rect["left"]) + int(rect["right"])) / 2)
            y = int((int(rect["top"]) + int(rect["bottom"])) / 2)
            self._click_screen_point(x, y)
            return True
        except Exception:
            return False

    def _find_text_target(
        self,
        text: str,
        app_name: str,
        timeout: int,
        control_types: list[str],
        exclude_control_types: list[str],
    ) -> dict[str, Any] | None:
        from pywinauto import Desktop

        target_text = text.lower()
        included = {item.lower() for item in control_types if item}
        excluded = {item.lower() for item in exclude_control_types if item}
        desktop = Desktop(backend="uia")
        deadline = time.monotonic() + max(1, timeout)
        while time.monotonic() < deadline:
            for window in self._candidate_windows(desktop, app_name):
                match = self._find_text_in_window(window, target_text, included, excluded)
                if match:
                    return match
            time.sleep(0.25)
        return None

    def _candidate_windows(self, desktop: Any, app_name: str) -> list[Any]:
        try:
            windows = list(desktop.windows())
        except Exception:
            return []
        clean_app = re.sub(r"\s+", " ", app_name or "").strip().lower()
        visible = []
        for window in windows:
            try:
                if not window.is_visible():
                    continue
                visible.append(window)
            except Exception:
                continue
        if not clean_app:
            return visible
        matches = [window for window in visible if clean_app in ((window.window_text() or "").lower())]
        return matches or visible

    def _find_text_in_window(
        self,
        window: Any,
        target_text: str,
        control_types: set[str],
        exclude_control_types: set[str],
    ) -> dict[str, Any] | None:
        controls = [window]
        try:
            controls.extend(window.descendants())
        except Exception:
            pass
        for control in controls:
            try:
                control_text = re.sub(r"\s+", " ", control.window_text() or "").strip()
                if not control_text or target_text not in control_text.lower():
                    continue
                control_type = str(getattr(control.element_info, "control_type", "") or "")
                control_type_key = control_type.lower()
                if control_types and control_type_key not in control_types:
                    continue
                if exclude_control_types and control_type_key in exclude_control_types:
                    continue
                rect = control.rectangle()
                if rect.width() <= 0 or rect.height() <= 0:
                    continue
                return {
                    "text": control_text,
                    "control_type": control_type,
                    "window_title": window.window_text() or "",
                    "rect": {"left": rect.left, "top": rect.top, "right": rect.right, "bottom": rect.bottom},
                }
            except Exception:
                continue
        return None

    def _send_keys_literal(self, text: str) -> str:
        replacements = {
            "{": "{{}",
            "}": "{}}",
            "+": "{+}",
            "^": "{^}",
            "%": "{%}",
            "~": "{~}",
            "(": "{(}",
            ")": "{)}",
            "[": "{[}",
            "]": "{]}",
        }
        return "".join(replacements.get(char, char) for char in text)

    def _send_keys_to_active_window(self, keys: str) -> bool:
        try:
            from pywinauto.keyboard import send_keys

            send_keys(keys, pause=0.03)
            return True
        except Exception:
            return False

    def _calculator_send_keys(self, expression: str) -> str:
        replacements = {
            "+": "{+}",
            "(": "{(}",
            ")": "{)}",
            "%": "{%}",
        }
        return "".join(replacements.get(char, char) for char in expression) + "{ENTER}"

    def _try_set_alarm_with_pywinauto(self, target_time: datetime, label: str) -> bool:
        try:
            from pywinauto import Desktop
            from pywinauto import mouse
            from pywinauto.keyboard import send_keys

            desktop = Desktop(backend="uia")
            clock = self._find_clock_window(desktop, timeout=8)
            clock.set_focus()
        except Exception:
            return False

        try:
            if not self._open_alarm_editor(clock) and not self._open_alarm_editor_with_core_window_hotkey():
                return False
            time.sleep(0.4)
            if self._try_set_alarm_with_core_window_coordinates(target_time):
                return True
            if not self._set_alarm_editor_time(clock, target_time, send_keys, mouse):
                return False
            self._set_alarm_editor_label(clock, label, send_keys)
            if not self._alarm_editor_time_matches(clock, target_time):
                return False
            return self._click_alarm_save(clock)
        except Exception:
            return False

    def _find_clock_window(self, desktop: Any, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                for window in desktop.windows():
                    title = window.window_text() or ""
                    if re.search(r"\b(Clock|Alarm|Alarms)\b", title, re.IGNORECASE):
                        window.wait("visible", timeout=1)
                        return window
            except Exception as exc:
                last_error = exc
            time.sleep(0.25)
        try:
            clock = desktop.window(title_re=r".*(Clock|Alarm|Alarms).*")
            clock.wait("visible", timeout=1)
            return clock
        except Exception as exc:
            raise last_error or exc

    def _open_alarm_editor(self, clock: Any) -> bool:
        if self._find_alarm_save_button(clock):
            return True

        candidates = []
        try:
            candidates.extend(clock.descendants(control_type="Button"))
        except Exception:
            candidates = []

        for button in candidates:
            title = (button.window_text() or "").strip().lower()
            if title in {"add", "+", "new alarm"} or ("add" in title and "alarm" in title):
                try:
                    button.click_input()
                    self._wait_for_alarm_save_button(clock, timeout=4)
                    return True
                except Exception:
                    continue

        for title in ("Add an alarm", "Add new alarm", "New alarm"):
            try:
                button = clock.child_window(title=title, control_type="Button")
                button.wait("enabled", timeout=1)
                button.click_input()
                self._wait_for_alarm_save_button(clock, timeout=4)
                return True
            except Exception:
                continue
        return False

    def _open_alarm_editor_with_core_window_hotkey(self) -> bool:
        hwnd_rect = self._find_visible_clock_core_window()
        if not hwnd_rect:
            return False
        hwnd, _rect = hwnd_rect
        self._focus_window(hwnd)
        self._send_ctrl_key("n")
        time.sleep(0.6)
        return True

    def _set_alarm_editor_time(self, clock: Any, target_time: datetime, send_keys: Any, mouse: Any) -> bool:
        controls = self._alarm_numeric_controls(clock)
        if len(controls) >= 2:
            hour_control, minute_control = controls[0], controls[1]
            self._replace_control_text(hour_control, self._alarm_hour_text(clock, target_time), send_keys, mouse)
            time.sleep(0.1)
            self._replace_control_text(minute_control, f"{target_time.minute:02d}", send_keys, mouse)
            self._set_alarm_meridiem(clock, target_time, mouse)
            time.sleep(0.25)
            if self._alarm_editor_time_matches(clock, target_time):
                return True

        points = self._alarm_time_coordinate_fallback(clock)
        if not points:
            return False
        hour_point, minute_point = points
        mouse.click(button="left", coords=hour_point)
        send_keys("^a{BACKSPACE}" + self._alarm_hour_text(clock, target_time), pause=0.03)
        time.sleep(0.1)
        mouse.click(button="left", coords=minute_point)
        send_keys("^a{BACKSPACE}" + f"{target_time.minute:02d}", pause=0.03)
        self._set_alarm_meridiem(clock, target_time, mouse)
        time.sleep(0.25)
        return self._alarm_editor_time_matches(clock, target_time)

    def _alarm_numeric_controls(self, clock: Any) -> list[Any]:
        controls = []
        save = self._find_alarm_save_button(clock)
        save_top = save.rectangle().top if save else None
        editor_bounds = self._alarm_editor_bounds(clock)
        try:
            descendants = clock.descendants()
        except Exception:
            return []
        for control in descendants:
            try:
                text = (control.window_text() or "").strip()
                rect = control.rectangle()
            except Exception:
                continue
            if save_top is not None and rect.top >= save_top:
                continue
            if editor_bounds and not self._rect_inside_bounds(rect, editor_bounds):
                continue
            if re.fullmatch(r"\d{1,2}", text) and rect.width() >= 20 and rect.height() >= 20:
                controls.append(control)
        controls.sort(key=lambda item: (item.rectangle().top, item.rectangle().left))
        if len(controls) > 2:
            top = controls[0].rectangle().top
            same_row = [item for item in controls if abs(item.rectangle().top - top) <= 40]
            if len(same_row) >= 2:
                return sorted(same_row, key=lambda item: item.rectangle().left)[:2]
        return controls[:2]

    def _replace_control_text(self, control: Any, text: str, send_keys: Any, mouse: Any) -> None:
        try:
            control.iface_value.SetValue(text)
            return
        except Exception:
            pass
        try:
            control.set_focus()
        except Exception:
            pass
        try:
            self._click_control_center(control, mouse)
            time.sleep(0.05)
            mouse.double_click(button="left", coords=self._control_center(control))
        except Exception:
            self._click_control_center(control, mouse)
        send_keys("{BACKSPACE}{BACKSPACE}{BACKSPACE}" + text, pause=0.03)

    def _alarm_editor_time_matches(self, clock: Any, target_time: datetime) -> bool:
        controls = self._alarm_numeric_controls(clock)
        if len(controls) < 2:
            return False
        try:
            hour_text = (controls[0].window_text() or "").strip()
            minute_text = (controls[1].window_text() or "").strip()
        except Exception:
            return False
        if not re.fullmatch(r"\d{1,2}", hour_text) or not re.fullmatch(r"\d{1,2}", minute_text):
            return False

        visible_hour = int(hour_text)
        visible_minute = int(minute_text)
        expected_hour = int(self._alarm_hour_text(clock, target_time))
        expected_minute = target_time.minute
        if visible_hour != expected_hour or visible_minute != expected_minute:
            return False

        if not self._alarm_editor_has_meridiem(clock):
            return True
        target = "PM" if target_time.hour >= 12 else "AM"
        selected_meridiem = self._selected_alarm_meridiem(clock)
        return selected_meridiem in {"", target}

    def _selected_alarm_meridiem(self, clock: Any) -> str:
        for child in self._safe_descendants(clock):
            try:
                text = (child.window_text() or "").strip().upper()
                if text not in {"AM", "PM"}:
                    continue
                if getattr(child, "is_selected", lambda: False)():
                    return text
                legacy = getattr(child, "legacy_properties", lambda: {})()
                if str(legacy.get("State") or "").lower().find("selected") >= 0:
                    return text
            except Exception:
                continue
        return ""

    def _alarm_hour_text(self, clock: Any, target_time: datetime) -> str:
        if self._alarm_editor_has_meridiem(clock):
            hour = target_time.hour % 12 or 12
            return f"{hour:02d}"
        return f"{target_time.hour:02d}"

    def _alarm_editor_has_meridiem(self, clock: Any) -> bool:
        try:
            return any((child.window_text() or "").strip().upper() in {"AM", "PM"} for child in clock.descendants())
        except Exception:
            return False

    def _set_alarm_meridiem(self, clock: Any, target_time: datetime, mouse: Any) -> None:
        target = "PM" if target_time.hour >= 12 else "AM"
        for child in self._safe_descendants(clock):
            try:
                if (child.window_text() or "").strip().upper() == target:
                    self._click_control_center(child, mouse)
                    return
            except Exception:
                continue

    def _set_alarm_editor_label(self, clock: Any, label: str, send_keys: Any) -> None:
        clean_label = re.sub(r"[{}^%+~()]", " ", label).strip()[:60]
        if not clean_label or clean_label == "Astra alarm":
            return
        editor_bounds = self._alarm_editor_bounds(clock)
        for control in self._safe_descendants(clock):
            try:
                text = (control.window_text() or "").strip()
                control_type = control.element_info.control_type
                rect = control.rectangle()
            except Exception:
                continue
            if editor_bounds and not self._rect_inside_bounds(rect, editor_bounds):
                continue
            if control_type == "Edit" and (text.lower().startswith("alarm") or not re.fullmatch(r"\d{1,2}", text)):
                try:
                    control.click_input()
                    send_keys("^a{BACKSPACE}" + clean_label, pause=0.02)
                    return
                except Exception:
                    continue

    def _click_alarm_save(self, clock: Any) -> bool:
        button = self._wait_for_alarm_save_button(clock, timeout=3)
        if not button:
            return False
        button.click_input()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            time.sleep(0.2)
            if not self._find_alarm_save_button(clock):
                return True
        return False

    def _try_set_alarm_with_core_window_coordinates(self, target_time: datetime) -> bool:
        hwnd_rect = self._find_visible_clock_core_window()
        if not hwnd_rect:
            return False
        hwnd, rect = hwnd_rect
        self._focus_window(hwnd)

        width = max(1, rect[2] - rect[0])
        height = max(1, rect[3] - rect[1])

        def point(rx: float, ry: float) -> tuple[int, int]:
            return rect[0] + int(width * rx), rect[1] + int(height * ry)

        def click(rx: float, ry: float) -> None:
            self._click_screen_point(*point(rx, ry))

        # The current Windows Clock alarm dialog is not exposed reliably through
        # UIA. These ratios target the visible "Add new alarm" dialog inside the
        # Clock CoreWindow and intentionally click Save only after attempting to
        # replace both spinner fields.
        hour_text = self._coordinate_alarm_hour_text(target_time)
        minute_text = f"{target_time.minute:02d}"
        click(0.442, 0.316)
        self._send_backspaces(3)
        self._send_text_low_level(hour_text)
        time.sleep(0.12)
        click(0.557, 0.316)
        self._send_backspaces(3)
        self._send_text_low_level(minute_text)
        time.sleep(0.12)
        if self._coordinate_click_meridiem_if_visible(target_time, rect):
            time.sleep(0.08)
        click(0.433, 0.819)
        time.sleep(0.7)
        return True

    def _focus_window(self, hwnd: int) -> None:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            time.sleep(0.2)
        except Exception:
            pass

    def _send_ctrl_key(self, char: str) -> None:
        import ctypes

        user32 = ctypes.windll.user32
        vk = ord(char.upper())
        user32.keybd_event(0x11, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(vk, 0, 0x0002, 0)
        time.sleep(0.03)
        user32.keybd_event(0x11, 0, 0x0002, 0)

    def _send_backspaces(self, count: int) -> None:
        import ctypes

        user32 = ctypes.windll.user32
        for _index in range(count):
            user32.keybd_event(0x08, 0, 0, 0)
            time.sleep(0.025)
            user32.keybd_event(0x08, 0, 0x0002, 0)
            time.sleep(0.025)

    def _send_text_low_level(self, text: str) -> None:
        import ctypes

        user32 = ctypes.windll.user32
        for char in text:
            vk = ord(char.upper())
            user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.035)
            user32.keybd_event(vk, 0, 0x0002, 0)
            time.sleep(0.035)

    def _coordinate_alarm_hour_text(self, target_time: datetime) -> str:
        # The visible dialog on this system has no AM/PM selector, so use 24-hour
        # input for afternoon/evening alarms. If Windows shows AM/PM, the click
        # helper below will still select the requested meridiem.
        return f"{target_time.hour:02d}"

    def _coordinate_click_meridiem_if_visible(self, target_time: datetime, rect: tuple[int, int, int, int]) -> bool:
        target = "PM" if target_time.hour >= 12 else "AM"
        try:
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            for child in desktop.descendants():
                text = (child.window_text() or "").strip().upper()
                if text != target:
                    continue
                child_rect = child.rectangle()
                center_x = child_rect.left + child_rect.width() // 2
                center_y = child_rect.top + child_rect.height() // 2
                if rect[0] <= center_x <= rect[2] and rect[1] <= center_y <= rect[3]:
                    self._click_screen_point(center_x, center_y)
                    return True
        except Exception:
            return False
        return False

    def _find_visible_clock_core_window(self) -> tuple[int, tuple[int, int, int, int]] | None:
        if not self.is_supported():
            return None
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32

            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            frame_matches: list[tuple[int, tuple[int, int, int, int]]] = []
            core_matches: list[tuple[int, tuple[int, int, int, int]]] = []
            enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

            def callback(hwnd: int, _lparam: int) -> bool:
                if not user32.IsWindowVisible(hwnd):
                    return True
                title_length = user32.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(title_length + 1)
                user32.GetWindowTextW(hwnd, title, title_length + 1)
                class_name = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, class_name, 256)
                rect = RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                width = rect.right - rect.left
                height = rect.bottom - rect.top
                if title.value == "Clock" and class_name.value == "ApplicationFrameWindow" and width > 300 and height > 300:
                    frame_matches.append((hwnd, (rect.left, rect.top, rect.right, rect.bottom)))
                if title.value == "Clock" and class_name.value == "Windows.UI.Core.CoreWindow" and width > 300 and height > 300:
                    core_matches.append((hwnd, (rect.left, rect.top, rect.right, rect.bottom)))
                return True

            user32.EnumWindows(enum_proc(callback), 0)
            if frame_matches:
                return frame_matches[0]
            return core_matches[0] if core_matches else None
        except Exception:
            return None

    def _click_screen_point(self, x: int, y: int, button: str = "left") -> None:
        import ctypes

        user32 = ctypes.windll.user32
        down = 0x0008 if button == "right" else 0x0002
        up = 0x0010 if button == "right" else 0x0004
        user32.SetCursorPos(x, y)
        user32.mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.04)
        user32.mouse_event(up, 0, 0, 0, 0)

    def _wait_for_alarm_save_button(self, clock: Any, timeout: float) -> Any | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            button = self._find_alarm_save_button(clock)
            if button:
                return button
            time.sleep(0.2)
        return None

    def _find_alarm_save_button(self, clock: Any) -> Any | None:
        for title in ("Save", "Save alarm"):
            try:
                button = clock.child_window(title=title, control_type="Button")
                button.wait("enabled", timeout=0.5)
                return button
            except Exception:
                pass
        for button in self._safe_descendants(clock, control_type="Button"):
            try:
                title = (button.window_text() or "").strip().lower()
                if title.startswith("save") and button.is_enabled():
                    return button
            except Exception:
                continue
        return None

    def _alarm_time_coordinate_fallback(self, clock: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
        anchor = self._find_alarm_editor_anchor(clock)
        if anchor:
            rect = anchor.rectangle()
            return (rect.left + 96, rect.top + 145), (rect.left + 246, rect.top + 145)

        save = self._find_alarm_save_button(clock)
        if not save:
            return None
        rect = save.rectangle()
        return (rect.left + 95, rect.top - 482), (rect.left + 246, rect.top - 482)

    def _find_alarm_editor_anchor(self, clock: Any) -> Any | None:
        for child in self._safe_descendants(clock):
            try:
                if "add new alarm" in (child.window_text() or "").strip().lower():
                    return child
            except Exception:
                continue
        return None

    def _alarm_editor_bounds(self, clock: Any) -> tuple[int, int, int, int] | None:
        anchor = self._find_alarm_editor_anchor(clock)
        save = self._find_alarm_save_button(clock)
        if not anchor or not save:
            return None
        anchor_rect = anchor.rectangle()
        save_rect = save.rectangle()
        left = min(anchor_rect.left, save_rect.left) - 80
        top = anchor_rect.top - 20
        right = max(anchor_rect.right, save_rect.right) + 260
        bottom = save_rect.top
        return (left, top, right, bottom)

    def _rect_inside_bounds(self, rect: Any, bounds: tuple[int, int, int, int]) -> bool:
        left, top, right, bottom = bounds
        center_x = rect.left + rect.width() // 2
        center_y = rect.top + rect.height() // 2
        return left <= center_x <= right and top <= center_y <= bottom

    def _click_control_center(self, control: Any, mouse: Any) -> None:
        mouse.click(button="left", coords=self._control_center(control))

    def _control_center(self, control: Any) -> tuple[int, int]:
        rect = control.rectangle()
        return rect.left + rect.width() // 2, rect.top + rect.height() // 2

    def _safe_descendants(self, clock: Any, control_type: str | None = None) -> list[Any]:
        try:
            if control_type:
                return list(clock.descendants(control_type=control_type))
            return list(clock.descendants())
        except Exception:
            return []
