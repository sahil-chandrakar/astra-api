import os
import sys
import asyncio
from pathlib import Path

from app.services.desktop import DesktopActionService, DesktopTarget


def test_file_explorer_opens_home_folder_with_windows_shell(monkeypatch):
    service = DesktopActionService()
    startfile_calls: list[str] = []
    popen_calls: list[list[str]] = []

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", lambda path: startfile_calls.append(path), raising=False)
    monkeypatch.setattr(service, "_popen", lambda argv: popen_calls.append(argv))

    service._open_app(DesktopTarget(("file explorer",), "app", "explorer.exe", "File Explorer"))

    assert startfile_calls == [str(Path.home())]
    assert popen_calls == []


def test_regular_app_uses_fixed_argv(monkeypatch):
    service = DesktopActionService()
    popen_calls: list[list[str]] = []

    monkeypatch.setattr(service, "_popen", lambda argv: popen_calls.append(argv))

    service._open_app(DesktopTarget(("notepad",), "app", "notepad.exe", "Notepad"))

    assert popen_calls == [["notepad.exe"]]


def test_detector_prefers_google_drive_over_google():
    service = DesktopActionService()

    assert service.detect("open google drive").label == "Google Drive"
    assert service.detect("open drive").label == "Google Drive"


def test_open_target_uses_exact_url_without_redetecting(monkeypatch):
    service = DesktopActionService()
    opened: list[tuple[str, int]] = []
    target = DesktopTarget(("google drive", "drive"), "url", "https://drive.google.com/", "Google Drive")

    monkeypatch.setattr("webbrowser.open", lambda url, new=0: opened.append((url, new)))

    result = asyncio.run(service.open_target(target))

    assert result.ok is True
    assert result.target == "Google Drive"
    assert opened == [("https://drive.google.com/", 2)]


def test_file_explorer_can_open_drive_root(monkeypatch, tmp_path):
    service = DesktopActionService()
    opened_locations: list[str] = []

    monkeypatch.setattr(service, "_file_explorer_location", lambda drive=None: str(tmp_path) if drive == "E" else str(Path.home()))
    monkeypatch.setattr(service, "_open_file_explorer_location", lambda location: opened_locations.append(location))

    result = asyncio.run(service.open_file_explorer("E"))

    assert result.ok is True
    assert result.target == "E: drive"
    assert opened_locations == [str(tmp_path)]
