import asyncio
import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.models import (
    ActionResult,
    AutomationCancelRequest,
    AutomationEngineStatus,
    AutomationRecipe,
    AutomationRecipeCreateRequest,
    AutomationRunRequest,
    ChatResponse,
    CommandRequest,
)
from app.services.automations import AutomationService
from app.services.commands import CommandService
from app.services.llm import LlmService
from app.services.openrpa import OpenRPAAdapter, OpenRPACancelledError, OpenRPARunResult


def build_service(tmp_path: Path) -> AutomationService:
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


def test_openrpa_detection_uses_env_override(tmp_path, monkeypatch):
    exe = tmp_path / "OpenRPA.exe"
    exe.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("ASTRA_OPENRPA_EXE", str(exe))

    adapter = OpenRPAAdapter(Settings(data_dir=str(tmp_path / "data")))
    status = adapter.status()

    assert status.installed is True
    assert status.configured_path == str(exe)


def test_openrpa_detection_reports_missing_install(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTRA_OPENRPA_EXE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "program-files"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "program-files-x86"))

    settings = Settings(data_dir=str(tmp_path / "data"), astra_openrpa_exe="")
    status = OpenRPAAdapter(settings).status()

    assert status.installed is False
    assert "OpenRPA.exe was not found" in status.message


def test_openrpa_command_builds_id_and_typed_inputs(tmp_path):
    exe = tmp_path / "OpenRPA.exe"
    exe.write_text("fake", encoding="utf-8")
    adapter = OpenRPAAdapter(Settings(data_dir=str(tmp_path / "data"), astra_openrpa_exe=str(exe)))
    recipe = AutomationRecipe(
        id="recipe-1",
        name="Invoice Export",
        prompt="export invoice",
        engine="openrpa",
        workflow_ref="workflow-123",
        workflow_ref_type="id",
        inputs=["customer", "limit", "dry_run"],
    )

    command = adapter.build_command(recipe, {"customer": "Astra", "limit": 5, "dry_run": True})

    assert command == [str(exe), "/WorkflowID", "workflow-123", "-customer", "Astra", "-limit", "5", "-dry_run", "true"]


def test_openrpa_command_rejects_unsafe_filename_reference(tmp_path):
    exe = tmp_path / "OpenRPA.exe"
    exe.write_text("fake", encoding="utf-8")
    adapter = OpenRPAAdapter(Settings(data_dir=str(tmp_path / "data"), astra_openrpa_exe=str(exe)))
    recipe = AutomationRecipe(
        id="recipe-1",
        name="Unsafe",
        prompt="unsafe",
        engine="openrpa",
        workflow_ref="..\\secret.xaml",
        workflow_ref_type="filename",
    )

    with pytest.raises(ValueError, match="parent directory"):
        adapter.build_command(recipe, {})


def test_openrpa_recipe_registration_blocks_sensitive_metadata(tmp_path):
    service = build_service(tmp_path)

    with pytest.raises(ValueError, match="safety policy"):
        service.create_recipe(
            AutomationRecipeCreateRequest(
                name="Enter password",
                prompt="enter password",
                engine="openrpa",
                workflow_ref="workflow-123",
                aliases=["login"],
            )
        )


def test_openrpa_recipe_registration_and_match(tmp_path):
    service = build_service(tmp_path)

    recipe = service.create_recipe(
        AutomationRecipeCreateRequest(
            name="Invoice Export",
            prompt="export invoices",
            engine="openrpa",
            workflow_ref="workflow-123",
            aliases=["export invoices", "invoice bot"],
            inputs=["customer"],
            risk="safe_confirm",
        )
    )
    matched = service.match_openrpa_recipe("please export invoices for May")

    assert recipe.engine == "openrpa"
    assert matched is not None
    assert matched.id == recipe.id


class FakeOpenRPA:
    def __init__(self, cancel: bool = False):
        self.cancel = cancel
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.cancel_seen = False

    def status(self):
        return AutomationEngineStatus(id="openrpa", label="OpenRPA", installed=True, configured_path="C:\\OpenRPA\\OpenRPA.exe", message="ok")

    def validate_recipe(self, recipe):
        if not recipe.workflow_ref:
            raise ValueError("workflow_ref required")

    def run(self, recipe, inputs, cancel_event: threading.Event, on_event=None):
        self.calls.append((recipe.id, dict(inputs or {})))
        if on_event:
            on_event("stdout", "fake workflow started", {"stream": "stdout"})
        if self.cancel:
            while not cancel_event.is_set():
                threading.Event().wait(0.03)
            self.cancel_seen = True
            raise OpenRPACancelledError("cancelled")
        return OpenRPARunResult(
            returncode=0,
            stdout="fake workflow complete",
            stderr="",
            elapsed_seconds=0.1,
            command=["OpenRPA.exe", "/WorkflowID", recipe.workflow_ref],
        )


@pytest.mark.asyncio
async def test_openrpa_automation_run_uses_adapter_and_streams_events(tmp_path):
    service = build_service(tmp_path)
    fake = FakeOpenRPA()
    service.openrpa = fake  # type: ignore[assignment]
    recipe = service.create_recipe(
        AutomationRecipeCreateRequest(
            name="Invoice Export",
            prompt="export invoices",
            engine="openrpa",
            workflow_ref="workflow-123",
            inputs=["customer"],
            risk="safe_auto",
        )
    )

    run = await service.start_run(AutomationRunRequest(prompt=" ", recipe_id=recipe.id, inputs={"customer": "Astra"}))
    finished = await wait_for_run(service, run.id, {"complete"})

    assert fake.calls == [(recipe.id, {"customer": "Astra"})]
    assert finished.result == "fake workflow complete"
    assert any(event.type == "openrpa_stdout" for event in finished.events)
    assert any(event.type == "openrpa_complete" for event in finished.events)


@pytest.mark.asyncio
async def test_openrpa_automation_cancel_sets_cancel_event(tmp_path):
    service = build_service(tmp_path)
    fake = FakeOpenRPA(cancel=True)
    service.openrpa = fake  # type: ignore[assignment]
    recipe = service.create_recipe(
        AutomationRecipeCreateRequest(
            name="Invoice Export",
            prompt="export invoices",
            engine="openrpa",
            workflow_ref="workflow-123",
            risk="safe_auto",
        )
    )

    run = await service.start_run(AutomationRunRequest(prompt=" ", recipe_id=recipe.id))
    await wait_for_run(service, run.id, {"running"})
    cancelled = await service.cancel_run(run.id, AutomationCancelRequest(note="Stop OpenRPA."))

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    finished = await wait_for_run(service, run.id, {"cancelled"})
    assert finished.status == "cancelled"
    for _ in range(60):
        if fake.cancel_seen:
            break
        await asyncio.sleep(0.03)
    assert fake.cancel_seen is True


class FakeAgentSystem:
    async def chat(self, text, mode, astra_pro=False):
        return ChatResponse(answer=f"chat: {text}")


class FakeDesktopActions:
    def detect(self, text):
        return None

    def plan_message(self, text):
        return ActionResult(ok=False, action="plan", target="", message="plan")


class FakeSafeAgent:
    async def handle_natural_language(self, text, confirmed=False):
        return None


@pytest.mark.asyncio
async def test_agent_mode_returns_openrpa_suggestion_without_starting_process(tmp_path):
    service = build_service(tmp_path)
    fake = FakeOpenRPA()
    service.openrpa = fake  # type: ignore[assignment]
    recipe = service.create_recipe(
        AutomationRecipeCreateRequest(
            name="Invoice Export",
            prompt="export invoices",
            engine="openrpa",
            workflow_ref="workflow-123",
            aliases=["export invoices"],
        )
    )
    command_service = CommandService(
        FakeAgentSystem(),  # type: ignore[arg-type]
        FakeDesktopActions(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        FakeSafeAgent(),  # type: ignore[arg-type]
        automation_service=service,
    )

    response = await command_service.handle(CommandRequest(text="please export invoices", mode="agents"))

    assert response.intent == "automation_suggestion"
    assert response.automation_suggestion is not None
    assert response.automation_suggestion.recipe.id == recipe.id
    assert fake.calls == []
