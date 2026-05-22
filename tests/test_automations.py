import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models import (
    AutomationCancelRequest,
    AutomationConfirmRequest,
    AutomationContinueRequest,
    AutomationRecipeCreateRequest,
    AutomationRun,
    AutomationRunRequest,
    LlmProfileConfig,
    LlmSettingsUpdateRequest,
)
from app.services.automations import AutomationCancelledError, AutomationService, VIDEO_DOWNLOAD_FALLBACK_FORMAT_SELECTOR
from app.services.llm import NVIDIA_PRO_MODELS, LlmService
from app.services.windows_automation import WindowsAutomationResult


def build_service(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    settings.cerebras_api_key = ""
    return AutomationService(settings, LlmService(settings))


class FakeWindowsDesktop:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def open_app(self, app_name: str):
        self.calls.append(("open_app", app_name))
        return WindowsAutomationResult(True, "app_opened", f"Opened {app_name}.", {"app": app_name})

    def find_text(self, text: str, app_name: str = "", timeout: int = 8, control_types=None, exclude_control_types=None):
        self.calls.append(("find_text", text))
        return WindowsAutomationResult(
            True,
            "desktop_text_found",
            f"Found '{text}'.",
            {
                "target": {
                    "text": text,
                    "control_type": "Text",
                    "window_title": app_name or "Desktop",
                    "rect": {"left": 10, "top": 10, "right": 80, "bottom": 40},
                }
            },
        )

    def click_target(self, target=None, text: str = "", button: str = "left"):
        self.calls.append(("click_target", text or str(target)))
        return WindowsAutomationResult(True, "desktop_clicked", "Clicked the desktop target.", {"target": target})

    def type_text(self, text: str, replace: bool = False):
        self.calls.append(("type_text", text))
        return WindowsAutomationResult(True, "desktop_typed", "Typed text into the focused desktop control.", {"text_length": len(text), "replace": replace})

    def press_key(self, key: str):
        self.calls.append(("press_key", key))
        return WindowsAutomationResult(True, "desktop_key_pressed", f"Pressed {key}.", {"key": key})

    def prepare_whatsapp_message(self, contact: str, message: str):
        self.calls.append(("prepare_whatsapp_message", f"{contact}:{message}"))
        return WindowsAutomationResult(True, "whatsapp_prepared", f"Prepared WhatsApp message to {contact}.", {"contact": contact, "message_length": len(message)})

    def send_prepared_whatsapp_message(self, contact: str = "", expected_message: str = ""):
        self.calls.append(("send_prepared_whatsapp_message", f"{contact}:{expected_message}"))
        return WindowsAutomationResult(True, "whatsapp_sent", "Sent the prepared WhatsApp message.", {"contact": contact, "message_length": len(expected_message)})

    def verify_text(self, text: str, app_name: str = "", timeout: int = 6, control_types=None, exclude_control_types=None):
        self.calls.append(("verify_text", text))
        return WindowsAutomationResult(
            True,
            "desktop_verified",
            f"Verified '{text}' is visible.",
            {"text": text, "app": app_name, "control_types": control_types or [], "exclude_control_types": exclude_control_types or []},
        )


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
    assert recipe.status == "executable"
    assert service.get_recipe(recipe.id) is not None
    assert service.delete_recipe(recipe.id) is True
    assert service.list_recipes() == []


@pytest.mark.asyncio
async def test_recipe_builder_uses_clarified_goal_not_wrapper_prompt(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt="Create a new reusable automation.", create_recipe=True))
    waiting = await wait_for_run(service, run.id, {"waiting_for_user"})

    assert "What kind of automation" in waiting.events[-1].message

    await service.continue_run(run.id, AutomationContinueRequest(note="open whatsapp app and send hiii to bebo2"))
    finished = await wait_for_run(service, run.id, {"complete"})
    recipes = service.list_recipes()

    assert finished.recipe_id
    assert recipes[0].prompt == "open whatsapp app and send hiii to bebo2"
    assert recipes[0].name == "Send WhatsApp message"
    assert recipes[0].status == "executable"
    assert recipes[0].missing_tools == []
    assert "Create a new reusable automation" not in recipes[0].prompt


@pytest.mark.asyncio
async def test_recipe_builder_saves_direct_whatsapp_goal_as_executable_desktop_recipe(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt='open whatsapp app and send message "hello" to bebo 2', create_recipe=True))
    finished = await wait_for_run(service, run.id, {"complete"})
    recipe = service.get_recipe(finished.recipe_id)

    assert recipe is not None
    assert recipe.status == "executable"
    assert recipe.risk == "safe_confirm"
    assert [step["tool"] for step in recipe.steps][:2] == ["app.resolve", "app.open"]
    assert "windows.send_prepared_whatsapp_message" in [step["tool"] for step in recipe.steps]
    assert "system.ask_user" not in [step["tool"] for step in recipe.steps]


@pytest.mark.asyncio
async def test_direct_message_prompt_uses_desktop_plan_before_llm(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps('open whatsapp app and send message "hello" to bebo 2')
    tools = [step["tool"] for step in steps]

    assert tools[:2] == ["app.resolve", "app.open"]
    assert "windows.prepare_whatsapp_message" in tools
    assert "windows.send_prepared_whatsapp_message" in tools
    assert tools[-1] == "desktop.verify_text"
    assert steps[-1]["args"]["exclude_control_types"] == ["Edit"]


@pytest.mark.asyncio
async def test_whatsapp_message_parser_strips_app_words_from_message(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open whatsapp and send whatsapp message hello to anurag 1")
    prepare_step = next(step for step in steps if step["tool"] == "windows.prepare_whatsapp_message")

    assert prepare_step["args"]["contact"] == "anurag 1"
    assert prepare_step["args"]["message"] == "hello"


@pytest.mark.asyncio
async def test_direct_message_run_waits_for_send_confirmation_before_pressing_enter(tmp_path):
    service = build_service(tmp_path)
    fake_windows = FakeWindowsDesktop()
    service.windows = fake_windows  # type: ignore[assignment]

    run = await service.start_run(AutomationRunRequest(prompt='open whatsapp app and send message "hello" to bebo 2'))
    waiting = await wait_for_run(service, run.id, {"confirmation_required"})

    assert waiting.confirmation is not None
    assert ("prepare_whatsapp_message", "bebo 2:hello") in fake_windows.calls
    assert ("send_prepared_whatsapp_message", "bebo 2:hello") not in fake_windows.calls

    await service.confirm_run(run.id, AutomationConfirmRequest(approved=True))
    finished = await wait_for_run(service, run.id, {"complete"})

    assert ("send_prepared_whatsapp_message", "bebo 2:hello") in fake_windows.calls
    assert ("verify_text", "hello") in fake_windows.calls
    assert finished.result == "Verified 'hello' is visible."


@pytest.mark.asyncio
async def test_non_executable_recipe_does_not_falsely_run(tmp_path):
    service = build_service(tmp_path)
    recipe = service.create_recipe(
        AutomationRecipeCreateRequest(
            name="Send WhatsApp message",
            prompt="open whatsapp app and send hiii to bebo2",
            steps=[{"tool": "desktop.find_text", "description": "Find contact.", "args": {"text": "bebo2"}}],
            status="needs_tools",
            missing_tools=["desktop.find_text"],
        )
    )

    run = await service.start_run(AutomationRunRequest(prompt=" ", recipe_id=recipe.id))
    finished = await wait_for_run(service, run.id, {"error"})

    assert "not executable yet" in finished.error


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

    assert steps[0]["tool"] == "youtube.search"
    assert steps[0]["args"]["query"] == "codewithharry"


@pytest.mark.asyncio
async def test_youtube_latest_name_prompt_uses_structured_youtube_tools(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and search for wwe latest video and just name the video and nothing else")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "wwe"
    assert steps[0]["args"]["action"] == "name"
    assert steps[0]["args"]["result_type"] == "video"
    assert steps[0]["args"]["upload_date"] == "recent"
    assert steps[0]["args"]["sort"] == "upload_date"
    assert steps[0]["args"]["route"] == "auto"
    assert steps[1]["args"]["mode"] == "name"


@pytest.mark.asyncio
async def test_youtube_play_and_filter_prompt_uses_play_result(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("play latest wwe live video on youtube")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "wwe"
    assert steps[0]["args"]["action"] == "play"
    assert steps[0]["args"]["result_type"] == "live"
    assert steps[0]["args"]["upload_date"] == "recent"
    assert steps[0]["args"]["route"] == "auto"
    assert steps[1]["args"]["mode"] == "play"


@pytest.mark.asyncio
async def test_youtube_explicit_channel_prompt_uses_channel_route(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and search for codewithharry channel newest video and just name the video")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "codewithharry"
    assert steps[0]["args"]["route"] == "channel"
    assert steps[0]["args"]["channel_hint"] is True
    assert steps[0]["args"]["result_type"] == "video"
    assert steps[0]["args"]["sort"] == "upload_date"
    assert steps[1]["args"]["mode"] == "name"


@pytest.mark.asyncio
async def test_youtube_topic_prompt_uses_topic_route(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and search for python programming latest video and just name the video")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "python programming"
    assert steps[0]["args"]["route"] == "topic"
    assert steps[0]["args"]["channel_hint"] is False
    assert steps[0]["args"]["sort"] == "upload_date"


@pytest.mark.asyncio
async def test_youtube_popular_channel_prompt_uses_view_count_sort(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("popular video from Veritasium channel on youtube just name the title")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "Veritasium"
    assert steps[0]["args"]["route"] == "channel"
    assert steps[0]["args"]["sort"] == "view_count"
    assert steps[1]["args"]["mode"] == "name"


@pytest.mark.asyncio
async def test_youtube_source_qualified_content_prompt_uses_target_search(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and play song from t series : asma ko chu kar dekha song")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "asma ko chu kar dekha song t series"
    assert steps[0]["args"]["channel_query"] == "t series"
    assert steps[0]["args"]["content_query"] == "asma ko chu kar dekha song"
    assert steps[0]["args"]["source_qualified"] is True
    assert steps[0]["args"]["route"] == "topic"
    assert steps[0]["args"]["action"] == "play"
    assert steps[1]["args"]["mode"] == "play"


@pytest.mark.asyncio
async def test_youtube_source_qualified_content_prompt_is_general(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and play the rust ownership lecture from freecodecamp: borrow checker explained")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "borrow checker explained freecodecamp"
    assert steps[0]["args"]["channel_query"] == "freecodecamp"
    assert steps[0]["args"]["content_query"] == "borrow checker explained"
    assert steps[0]["args"]["source_qualified"] is True
    assert steps[0]["args"]["route"] == "topic"
    assert steps[0]["args"]["action"] == "play"


@pytest.mark.asyncio
async def test_youtube_source_qualified_play_from_prompt_without_media_word(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and play python decorators tutorial from freecodecamp")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "python decorators tutorial freecodecamp"
    assert steps[0]["args"]["channel_query"] == "freecodecamp"
    assert steps[0]["args"]["content_query"] == "python decorators tutorial"
    assert steps[0]["args"]["source_qualified"] is True
    assert steps[0]["args"]["route"] == "topic"
    assert steps[0]["args"]["action"] == "play"


@pytest.mark.asyncio
async def test_youtube_source_qualified_about_prompt(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and search for latest video by NASA about mars rover and just name the video")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "mars rover NASA"
    assert steps[0]["args"]["channel_query"] == "NASA"
    assert steps[0]["args"]["content_query"] == "mars rover"
    assert steps[0]["args"]["source_qualified"] is True
    assert steps[0]["args"]["route"] == "topic"
    assert steps[0]["args"]["action"] == "name"


@pytest.mark.asyncio
async def test_youtube_channel_url_prompt_uses_channel_handle(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("open youtube and search for youtube.com/@veritasium latest video and just name the video")

    assert [step["tool"] for step in steps] == ["youtube.search", "youtube.result"]
    assert steps[0]["args"]["query"] == "veritasium"
    assert steps[0]["args"]["route"] == "channel"
    assert steps[0]["args"]["channel_hint"] is True
    assert steps[0]["args"]["action"] == "name"


def test_youtube_channel_scoring_handles_spaced_compound_names(tmp_path):
    service = build_service(tmp_path)

    score = service._score_youtube_channel_candidate(
        "mr beast",
        {"title": "MrBeast", "handle": "@MrBeast", "url": "https://www.youtube.com/@MrBeast"},
    )

    assert score == 1.0


def test_youtube_title_filter_rejects_live_badges_and_durations(tmp_path):
    service = build_service(tmp_path)

    assert service._bad_youtube_title("LIVE") is True
    assert service._bad_youtube_title("7:33:01") is True


def test_source_qualified_youtube_results_prefer_matching_source(tmp_path):
    service = build_service(tmp_path)

    ranked = service._rank_youtube_source_results(
        {"channel_query": "freecodecamp", "content_query": "borrow checker explained", "source_qualified": True},
        [
            {
                "title": "Borrow Checker Explained Clearly",
                "channel": "Random Tutorials",
                "metadata": "",
                "url": "https://www.youtube.com/watch?v=wrong",
                "type": "video",
            },
            {
                "title": "Rust Ownership and Borrow Checker Explained",
                "channel": "freeCodeCamp.org",
                "metadata": "",
                "url": "https://www.youtube.com/watch?v=right",
                "type": "video",
            },
        ],
    )

    assert ranked[0]["url"] == "https://www.youtube.com/watch?v=right"


@pytest.mark.asyncio
async def test_planner_uses_configured_pro_profile_for_llm_fallback(tmp_path):
    service = build_service(tmp_path)
    service.settings.nvidia_api_key = "configured"
    service.llm.update_settings(
        LlmSettingsUpdateRequest(
            profiles={
                "fast": LlmProfileConfig(provider="cerebras", model="llama3.1-8b"),
                "pro": LlmProfileConfig(provider="nvidia", model=NVIDIA_PRO_MODELS[0]),
            }
        )
    )
    called_models = []

    async def fake_complete(_system_prompt, _user_prompt, model=None):
        called_models.append(model)
        return '{"steps":[{"tool":"browser.search","description":"Search the web.","args":{"site":"google","query":"ai automation"}}]}', []

    service.llm.complete = fake_complete

    steps = await service._plan_steps("research a safe browser automation flow for ai automation")

    assert called_models == [f"nvidia:{NVIDIA_PRO_MODELS[0]}"]
    assert steps[0]["tool"] == "browser.search"


def test_youtube_download_plan_opens_result_before_official_download(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("open youtube and search for codewithharry and download it's first latest video")

    assert [step["tool"] for step in steps] == ["youtube.search", "browser.click", "video.download_permitted"]
    assert steps[0]["args"]["query"] == "codewithharry"
    assert steps[0]["args"]["upload_date"] == "recent"


def test_direct_youtube_download_plan_uses_video_url_without_search(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("download this video https://www.youtube.com/watch?v=MhUS3zJ6WMs")

    assert [step["tool"] for step in steps] == ["video.download_permitted"]
    assert steps[0]["args"]["url"] == "https://www.youtube.com/watch?v=MhUS3zJ6WMs"


def test_permitted_public_youtube_link_download_skips_browser_open(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("Download this permitted public video: https://www.youtube.com/watch?v=i9_lboy-Et0")

    assert [step["tool"] for step in steps] == ["video.download_permitted"]
    assert steps[0]["args"]["url"] == "https://www.youtube.com/watch?v=i9_lboy-Et0"


@pytest.mark.asyncio
async def test_runtime_plans_downloaded_media_reference_without_phrase_router(tmp_path):
    service = build_service(tmp_path)

    steps = await service._plan_steps("play that downloaded song in vlc")

    assert [step["tool"] for step in steps] == ["artifact.resolve_reference", "app.open_with_file"]
    assert steps[0]["args"]["media_types"] == ["audio", "video"]
    assert steps[1]["args"]["app_name"] == "vlc"


def test_artifact_memory_resolves_latest_downloaded_media(tmp_path):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("artifact-memory-test")
    media_path = run_dir / "Licensed Demo Song.mp4"
    media_path.write_text("demo", encoding="utf-8")

    artifact = service.artifacts.record_download(media_path, "artifact-memory-test", title="Licensed Demo Song")
    resolved, matches, reason = service.artifacts.resolve_reference("play that downloaded song in vlc")

    assert resolved is not None
    assert resolved.id == artifact.id
    assert matches
    assert reason == ""


@pytest.mark.asyncio
async def test_artifact_resolve_then_open_with_requested_app(tmp_path):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("open-artifact-test")
    media_path = run_dir / "Lecture Clip.mp4"
    media_path.write_text("demo", encoding="utf-8")
    service.artifacts.record_download(media_path, "open-artifact-test", title="Lecture Clip")
    run = AutomationRun(id="open-artifact-test", prompt="play that downloaded video in vlc", status="running")
    service.runs[run.id] = run
    opened: list[tuple[str, str]] = []

    def fake_open(artifact, app_name):
        opened.append((artifact.filename, app_name))
        return {"artifact_id": artifact.id, "filename": artifact.filename, "path": artifact.path, "app": {"app_name": app_name, "path": "vlc.exe", "launch_mode": "executable"}}

    service.runtime.open_artifact_with_app = fake_open  # type: ignore[method-assign]

    await service._tool_artifact_resolve_reference(run, {"query": run.prompt, "media_types": ["video", "audio"]})
    await service._tool_app_open_with_file(run, {"app_name": "vlc"})

    assert opened == [("Lecture Clip.mp4", "vlc")]
    assert run.result == "Opened Lecture Clip.mp4 with vlc."
    assert [event.type for event in run.events[-2:]] == ["artifact_resolved", "app_opened"]


@pytest.mark.asyncio
async def test_runtime_loop_opens_latest_artifact_containing_folder(tmp_path):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("folder-open-test")
    media_path = run_dir / "Latest Clip.mp4"
    media_path.write_text("demo", encoding="utf-8")
    service.artifacts.record_download(media_path, "folder-open-test", title="Latest Clip")
    opened: list[str] = []

    def fake_open_folder(artifact):
        opened.append(artifact.filename)
        return {"artifact_id": artifact.id, "filename": artifact.filename, "path": artifact.path, "folder_path": str(Path(artifact.path).parent)}

    service.runtime.open_artifact_containing_folder = fake_open_folder  # type: ignore[method-assign]

    run = await service.start_run(AutomationRunRequest(prompt="open folder where file is located"))
    finished = await wait_for_run(service, run.id, {"complete"})

    assert opened == ["Latest Clip.mp4"]
    assert finished.result == "Opened the folder containing Latest Clip.mp4."
    assert any(event.type == "folder_opened" for event in finished.events)
    assert any(event.type == "verified" for event in finished.events)


@pytest.mark.asyncio
async def test_runtime_loop_waits_for_artifact_selection_then_resumes(tmp_path):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("folder-pick-test")
    first = run_dir / "Demo Clip.mp4"
    second = run_dir / "Demo Clip.mkv"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    first_artifact = service.artifacts.record_download(first, "folder-pick-test", title="Demo Clip")
    second_artifact = service.artifacts.record_download(second, "folder-pick-test", title="Demo Clip")
    opened: list[str] = []

    def fake_open_folder(artifact):
        opened.append(artifact.id)
        return {"artifact_id": artifact.id, "filename": artifact.filename, "path": artifact.path, "folder_path": str(Path(artifact.path).parent)}

    service.runtime.open_artifact_containing_folder = fake_open_folder  # type: ignore[method-assign]

    run = await service.start_run(AutomationRunRequest(prompt="open folder for demo file"))
    waiting = await wait_for_run(service, run.id, {"waiting_for_user"})

    assert waiting.agent_state["candidates"]
    assert opened == []

    await service.continue_run(run.id, AutomationContinueRequest(selected_artifact_id=first_artifact.id, note="use the mp4"))
    finished = await wait_for_run(service, run.id, {"complete"})

    assert opened == [first_artifact.id]
    assert finished.agent_state["last_artifact_id"] == first_artifact.id
    assert second_artifact.id != first_artifact.id


@pytest.mark.asyncio
async def test_continue_without_clarification_does_not_leave_runtime_stuck(tmp_path):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("folder-wait-test")
    for name in ("Demo Clip.mp4", "Demo Clip.mkv"):
        path = run_dir / name
        path.write_text("demo", encoding="utf-8")
        service.artifacts.record_download(path, "folder-wait-test", title="Demo Clip")

    run = await service.start_run(AutomationRunRequest(prompt="open folder for demo file"))
    waiting = await wait_for_run(service, run.id, {"waiting_for_user"})
    assert waiting.status == "waiting_for_user"

    await service.continue_run(run.id, AutomationContinueRequest())
    still_waiting = await wait_for_run(service, run.id, {"waiting_for_user"})

    assert still_waiting.status == "waiting_for_user"
    assert still_waiting.agent_state["candidates"]


@pytest.mark.asyncio
async def test_runtime_loop_replans_after_failed_tool_action(tmp_path):
    service = build_service(tmp_path)
    run_dir = service._safe_run_download_dir("folder-replan-test")
    media_path = run_dir / "Retry Clip.mp4"
    media_path.write_text("demo", encoding="utf-8")
    service.artifacts.record_download(media_path, "folder-replan-test", title="Retry Clip")
    calls = 0

    def flaky_open_folder(artifact):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Explorer was busy")
        return {"artifact_id": artifact.id, "filename": artifact.filename, "path": artifact.path, "folder_path": str(Path(artifact.path).parent)}

    service.runtime.open_artifact_containing_folder = flaky_open_folder  # type: ignore[method-assign]

    run = await service.start_run(AutomationRunRequest(prompt="open folder where file is located"))
    finished = await wait_for_run(service, run.id, {"complete"})

    assert calls == 2
    assert any(event.type == "replanned" for event in finished.events)


def test_alarm_plan_uses_windows_tool_not_google(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    monkeypatch.setattr(service, "_now", lambda: datetime(2026, 5, 20, 10, 0))

    steps = service._heuristic_plan("open alarm and set it for 5:30 PM today")

    assert [step["tool"] for step in steps] == ["windows.set_alarm"]
    assert steps[0]["args"]["target_iso"].startswith("2026-05-20T17:30")
    assert "error" not in steps[0]["args"]


def test_alarm_plan_rejects_past_today_time(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    monkeypatch.setattr(service, "_now", lambda: datetime(2026, 5, 20, 18, 0))

    steps = service._heuristic_plan("open alarm and set it for 5:30 PM today")

    assert [step["tool"] for step in steps] == ["windows.set_alarm"]
    assert "already passed" in steps[0]["args"]["error"]


def test_calculator_plan_sanitizes_expression(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("open calculator and calculate 45*6")

    assert [step["tool"] for step in steps] == ["windows.open_calculator"]
    assert steps[0]["args"]["expression"] == "45*6"


def test_file_explorer_plan_resolves_downloads_folder(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("open downloads folder")

    assert [step["tool"] for step in steps] == ["windows.open_file_explorer"]
    assert steps[0]["args"]["target"] == "downloads"


def test_file_explorer_plan_resolves_drive_target(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("open D drive in file explorer")

    assert [step["tool"] for step in steps] == ["windows.open_file_explorer"]
    assert steps[0]["args"]["target"] == "D:\\"


def test_destructive_file_prompt_uses_blocked_windows_tool(tmp_path):
    service = build_service(tmp_path)

    steps = service._heuristic_plan("delete this file")

    assert [step["tool"] for step in steps] == ["windows.reject_destructive_file_action"]


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


def test_video_download_does_not_require_rights_confirmation(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="rights-test", prompt="download video")
    step = {"tool": "video.download_permitted", "description": "Download video.", "args": {"url": "https://www.youtube.com/watch?v=abc123"}}

    assert service._requires_confirmation(run, step) is False


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
    assert any(event.type == "artifact_recorded" for event in run.events)
    assert any(event.type == "download_progress" and event.data["progress_percent"] == 100 for event in run.events)
    assert all("folder_path" in event.data for event in run.events if event.type.startswith("download_"))
    artifact = service.artifacts.resolve_reference("latest downloaded video")[0]
    assert artifact is not None
    assert artifact.filename == "Licensed Demo [abc123].mp4"


@pytest.mark.asyncio
async def test_blocked_prompt_finishes_with_error(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt="steal my password"))
    finished = await wait_for_run(service, run.id, {"error"})

    assert "blocked" in finished.error.lower()
    assert finished.events[-1].type == "error"


@pytest.mark.asyncio
async def test_windows_calculator_tool_emits_calculator_event(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="calculator-test", prompt="open calculator")

    class FakeWindows:
        def open_calculator(self, expression):
            return WindowsAutomationResult(True, "calculator", f"Opened Calculator and typed {expression}.", {"expression": expression})

    service.windows = FakeWindows()  # type: ignore[assignment]

    await service._tool_windows_open_calculator(run, {"expression": "45*6"})

    assert run.result == "Opened Calculator and typed 45*6."
    assert run.events[-1].type == "calculator"
    assert run.events[-1].data["expression"] == "45*6"


@pytest.mark.asyncio
async def test_windows_file_explorer_tool_emits_file_explorer_event(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="explorer-test", prompt="open downloads folder")

    class FakeWindows:
        def open_file_explorer(self, target):
            return WindowsAutomationResult(True, "file_explorer", "Opened downloads.", {"target": target, "opened_path": "C:\\Users\\Astra\\Downloads"})

    service.windows = FakeWindows()  # type: ignore[assignment]

    await service._tool_windows_open_file_explorer(run, {"target": "downloads"})

    assert run.result == "Opened downloads."
    assert run.events[-1].type == "file_explorer"
    assert "folder_path" not in run.events[-1].data


@pytest.mark.asyncio
async def test_windows_alarm_tool_emits_alarm_set_event(tmp_path, monkeypatch):
    service = build_service(tmp_path)
    monkeypatch.setattr(service, "_now", lambda: datetime(2026, 5, 20, 10, 0))
    run = AutomationRun(id="alarm-test", prompt="set alarm")

    class FakeWindows:
        def set_alarm(self, target_time, label):
            return WindowsAutomationResult(True, "alarm_set", "Set alarm for 5:30 PM.", {"alarm_time": target_time.isoformat(), "label": label})

    service.windows = FakeWindows()  # type: ignore[assignment]

    await service._tool_windows_set_alarm(run, {"target_iso": "2026-05-20T17:30:00", "label": "Astra alarm"})

    assert run.result == "Set alarm for 5:30 PM."
    assert run.events[-1].type == "alarm_set"


@pytest.mark.asyncio
async def test_past_alarm_tool_blocks_without_error_status(tmp_path):
    service = build_service(tmp_path)
    run = AutomationRun(id="past-alarm-test", prompt="set past alarm")

    await service._tool_windows_set_alarm(
        run,
        {"error": "That alarm time has already passed today, so Astra did not set an alarm. Choose a future time such as tomorrow."},
    )

    assert run.status == "cancelled"
    assert run.events[-1].type == "blocked"
    assert "did not set an alarm" in run.result


@pytest.mark.asyncio
async def test_destructive_file_run_is_cancelled_with_blocked_event(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt="move this file"))
    finished = await wait_for_run(service, run.id, {"cancelled"})

    assert any(event.type == "blocked" for event in finished.events)
    assert "does not delete" in finished.result


@pytest.mark.asyncio
async def test_python_recipe_builder_saves_executable_recipe_without_running_it(tmp_path):
    service = build_service(tmp_path)

    run = await service.start_run(AutomationRunRequest(prompt="create reusable automation with python", create_recipe=True))
    finished = await wait_for_run(service, run.id, {"complete"})
    recipe = service.get_recipe(finished.recipe_id)

    assert finished.result
    assert recipe is not None
    assert recipe.status == "executable"
    assert recipe.steps[0]["tool"] == "python.run_safe"
