import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models import AutomationCancelRequest, AutomationConfirmRequest, AutomationRecipeCreateRequest, AutomationRun, AutomationRunRequest
from app.services.automations import AutomationCancelledError, AutomationService, VIDEO_DOWNLOAD_FALLBACK_FORMAT_SELECTOR
from app.services.llm import LlmService


def build_service(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    settings.cerebras_api_key = ""
    return AutomationService(settings, LlmService(settings))


async def wait_for_run(service: AutomationService, run_id: str, statuses: set[str], timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        run = service.get_run(run_id)
        if run and run.status in statuses:
            return run
        await asyncio.sleep(0.03)
    run = service.get_run(run_id)
    raise AssertionError(f"Run {run_id} did not reach {statuses}; status={run.status if run else None}")


def test_recipe_crud(tmp_path):
    service = build_service(tmp_path)

    recipe = service.create_recipe(AutomationRecipeCreateRequest(name="Daily Search", prompt="search ai news", steps=[]))

    assert service.list_recipes()[0].id == recipe.id
    assert service.get_recipe(recipe.id) is not None
    assert service.delete_recipe(recipe.id) is True
    assert service.list_recipes() == []


def test_sanitize_steps_keeps_only_allowed_tools_and_safe_python(tmp_path):
    service = build_service(tmp_path)

    steps = service._sanitize_steps(
        [
            {"tool": "browser.search", "description": "search", "args": {"query": "hello", "site": "google"}},
            {"tool": "shell.exec", "description": "bad", "args": {"cmd": "del *"}},
            {"tool": "python.run_safe", "description": "bad python", "args": {"code": "import os\nprint(os.listdir('/'))"}},
            {"tool": "python.run_safe", "description": "youtube bypass", "args": {"code": "import yt_dlp\nyt_dlp.YoutubeDL({})"}},
        ]
    )

    assert [step["tool"] for step in steps] == ["browser.search"]


def test_extract_json_handles_empty_llm_output(tmp_path):
    service = build_service(tmp_path)

    assert service._extract_json(None) == {}


@pytest.mark.asyncio
async def test_planner_falls_back_when_llm_returns_empty_content(tmp_path):
    service = build_service(tmp_path)
    service.settings.cerebras_api_key = "configured"

    async def fake_complete(*_args, **_kwargs):
        return None, []

    service.llm.complete = fake_complete

    steps = await service._plan_steps("open youtube and search for codewithharry")

    assert steps[0]["tool"] == "browser.search"
    assert steps[0]["args"]["site"] == "youtube"


def test_youtube_download_plan_opens_result_before_official_download(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("open youtube and search for codewithharry and download it's first latest video")

    assert [step["tool"] for step in steps] == ["browser.search", "browser.click", "video.download_permitted"]
    assert steps[0]["args"]["query"] == "codewithharry latest"


def test_direct_youtube_download_plan_uses_video_url_without_search(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("download this video https://www.youtube.com/watch?v=MhUS3zJ6WMs")

    assert [step["tool"] for step in steps] == ["browser.open", "video.download_permitted"]
    assert steps[0]["args"]["url"] == "https://www.youtube.com/watch?v=MhUS3zJ6WMs"
    assert steps[1]["args"]["url"] == "https://www.youtube.com/watch?v=MhUS3zJ6WMs"


def test_download_unavailable_message_includes_youtube_link(tmp_path):
    service = build_service(tmp_path)

    message = service._download_unavailable_message("https://www.youtube.com/watch?v=abc123")

    assert "No official browser download control" in message
    assert "https://www.youtube.com/watch?v=abc123" in message


def test_video_info_blocks_unlisted_youtube(tmp_path):
    service = build_service(tmp_path)

    with pytest.raises(ValueError, match="unlisted"):
        service._validate_video_info({"availability": "unlisted"}, "https://www.youtube.com/watch?v=abc123", 500)


def test_safe_run_download_dir_stays_inside_downloads(tmp_path):
    service = build_service(tmp_path)

    run_dir = service._safe_run_download_dir("../escape")

    assert service.downloads_dir.resolve() in run_dir.parents


def test_parse_ytdlp_progress_line(tmp_path):
    service = build_service(tmp_path)

    progress = service._parse_ytdlp_progress_line("ASTRA_PROGRESS: 42.7%|1024|2048|NA")

    assert progress == {
        "status": "downloading",
        "progress_percent": 42,
        "downloaded_bytes": 1024,
        "total_bytes": 2048,
    }


def test_ytdlp_error_summary_keeps_relevant_lines(tmp_path):
    service = build_service(tmp_path)

    summary = service._summarize_ytdlp_failure(
        "\n".join(
            [
                "[download] 10%",
                "[download] Got error: HTTP Error 403: Forbidden. Retrying fragment 107 (10/10)...",
                "ERROR: The downloaded file is empty",
            ]
        )
    )

    assert "HTTP Error 403" in summary
    assert "downloaded file is empty" in summary


def test_ytdlp_command_prefers_top_quality_with_ffmpeg(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("quality-test")
    monkeypatch.setattr(service, "_ffmpeg_location", lambda: "C:\\tools\\ffmpeg.exe")

    command = service._ytdlp_download_command("https://www.youtube.com/watch?v=abc123", run_dir, 500)

    format_index = command.index("-f") + 1
    sort_index = command.index("--format-sort") + 1
    assert "bv*" in command[format_index]
    assert "height<=720" not in command[format_index]
    assert command[sort_index] == "proto:https,res,fps"
    assert "--merge-output-format" in command
    assert command[command.index("--ffmpeg-location") + 1] == "C:\\tools\\ffmpeg.exe"


def test_ytdlp_fallback_command_uses_progressive_mp4_without_ffmpeg(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("fallback-test")
    monkeypatch.setattr(service, "_ffmpeg_location", lambda: "C:\\tools\\ffmpeg.exe")

    command = service._ytdlp_download_command(
        "https://www.youtube.com/watch?v=abc123",
        run_dir,
        500,
        format_selector=VIDEO_DOWNLOAD_FALLBACK_FORMAT_SELECTOR,
        merge_output_format="",
    )

    assert command[command.index("-f") + 1].startswith("18/22")
    assert "--merge-output-format" not in command
    assert "--ffmpeg-location" not in command


def test_ytdlp_download_retries_progressive_after_high_quality_failure(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="retry-test", prompt="download video")
    run_dir = service._safe_run_download_dir(run.id)
    calls: list[list[str]] = []

    def fake_once(_run_id, command, _run_download_dir, _progress_hook):
        calls.append(command)
        if len(calls) == 1:
            (run_dir / "partial-high.webm").write_text("partial", encoding="utf-8")
            raise ValueError("ERROR: unable to download video data: HTTP Error 403: Forbidden")
        (run_dir / "fallback.mp4").write_text("ok", encoding="utf-8")

    service._run_ytdlp_command_once = fake_once

    service._run_ytdlp_download_subprocess(run, "https://www.youtube.com/watch?v=abc123", run_dir, set(), 500, lambda _progress: None)

    assert len(calls) == 2
    assert "bv*" in calls[0][calls[0].index("-f") + 1]
    assert calls[1][calls[1].index("-f") + 1].startswith("18/22")
    assert not (run_dir / "partial-high.webm").exists()
    assert (run_dir / "fallback.mp4").exists()
    assert any(event.type == "download_retry" for event in run.events)


def test_ytdlp_download_cancel_cleans_partial_files(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="cancel-cleanup-test", prompt="download video")
    run_dir = service._safe_run_download_dir(run.id)
    service._cancel_event_for_run(run.id).set()
    (run_dir / "partial-download.part").write_text("partial", encoding="utf-8")

    with pytest.raises(AutomationCancelledError):
        service._run_ytdlp_download_subprocess(run, "https://www.youtube.com/watch?v=abc123", run_dir, set(), 500, lambda _progress: None)

    assert not (run_dir / "partial-download.part").exists()


@pytest.mark.asyncio
async def test_cancel_run_kills_active_download_process(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="cancel-process-test", prompt="download video", status="running", current_url="https://www.youtube.com/watch?v=abc123")
    service.runs[run.id] = run

    class FakeProcess:
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    process = FakeProcess()
    service._active_download_processes[run.id] = process  # type: ignore[assignment]

    cancelled = await service.cancel_run(run.id, AutomationCancelRequest(note="Stop download."))

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert process.killed is True
    assert service._is_cancelled(run.id)
    assert any(event.type == "download_cancelled" for event in cancelled.events)


def test_selected_video_size_sums_merged_downloads(tmp_path):
    service = build_service(tmp_path)

    size = service._selected_video_size(
        {
            "requested_downloads": [
                {"filesize": 20},
                {"filesize_approx": 30},
            ]
        }
    )

    assert size == 50


def test_open_download_path_only_allows_download_folder(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    opened: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("os.startfile", lambda path: opened.append(path), raising=False)
    allowed = service._safe_run_download_dir("open-test")

    assert service.open_download_path(str(allowed)) is True
    assert service.open_download_path(str(tmp_path)) is False
    assert opened == [str(allowed)]


@pytest.mark.asyncio
async def test_video_confirmation_requires_rights_confirmation(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="rights-test", prompt="download video", confirmation={"kind": "video_download_permission"})
    service.runs[run.id] = run
    future = asyncio.get_running_loop().create_future()
    service._confirmation_futures[run.id] = future

    await service.confirm_run(run.id, AutomationConfirmRequest(approved=True, confirmed_rights=False))

    assert future.result() is False


@pytest.mark.asyncio
async def test_mocked_ytdlp_video_download_completes(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    run = AutomationRun(id="video-test", prompt="download video")
    service.runs[run.id] = run

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download=False):
            assert download is False
            return {
                "title": "Licensed Demo",
                "id": "abc123",
                "availability": "public",
                "filesize": 1024,
                "channel": "Astra",
            }

    def fake_subprocess(_run, _source_url, run_download_dir, _before_files, _max_size_bytes, progress_hook):
        output = run_download_dir / "Licensed Demo [abc123].mp4"
        output.write_text("demo", encoding="utf-8")
        progress_hook({"status": "downloading", "downloaded_bytes": 1024, "total_bytes": 1024})

    monkeypatch.setattr(service, "_run_ytdlp_download_subprocess", fake_subprocess)
    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))

    await service._tool_video_download_permitted(run, {"url": "https://www.youtube.com/watch?v=abc123"})

    assert run.result == "Downloaded Licensed Demo [abc123].mp4."
    assert any(event.type == "download_complete" for event in run.events)
    assert any(event.type == "download_progress" and event.data["progress_percent"] == 100 for event in run.events)
    assert all("folder_path" in event.data for event in run.events if event.type.startswith("download_"))


@pytest.mark.asyncio
async def test_blocked_prompt_finishes_with_error(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt="steal my password"))
    finished = await wait_for_run(service, run.id, {"error"})

    assert "blocked" in finished.error.lower()
    assert finished.events[-1].type == "error"


@pytest.mark.asyncio
async def test_python_step_requires_confirmation_then_completes_and_saves_recipe(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt="create reusable automation with python", create_recipe=True))
    waiting = await wait_for_run(service, run.id, {"confirmation_required"})

    assert waiting.confirmation is not None
    assert "Python" in waiting.confirmation["message"]

    await service.confirm_run(run.id, AutomationConfirmRequest(approved=True))
    finished = await wait_for_run(service, run.id, {"complete"})

    assert finished.result
    assert finished.recipe_id
    assert service.list_recipes()
