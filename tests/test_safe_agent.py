from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from app.config import Settings
from app.models import ActionResult, AgentCommandRequest
from app.services.desktop import DesktopActionService
from app.services.documents import DocumentService
from app.services.llm import LlmService
from app.services.memory import MemoryService
from app.services.reports import ReportService
from app.services.safe_agent import SafeAgentService
from app.services.study import StudyService
from app.services.voice import VoiceService


class RecordingDesktopActionService(DesktopActionService):
    def __init__(self):
        self.executed: list[str] = []

    async def execute(self, text: str) -> ActionResult:
        self.executed.append(text)
        target = self.detect(text)
        label = target.label if target else text
        return ActionResult(ok=True, action="open", target=label, message=f"Opened {label}.")

    async def open_target(self, target) -> ActionResult:
        self.executed.append(f"open {target.label}")
        return ActionResult(ok=True, action="open", target=target.label, message=f"Opened {target.label}.")

    async def open_file_explorer(self, drive: str | None = None) -> ActionResult:
        label = f"{drive.upper()}: drive" if drive else "File Explorer"
        self.executed.append(f"open {label}")
        return ActionResult(ok=True, action="open", target=label, message=f"Opened {label}.")


class FakeLlmService(LlmService):
    def __init__(self, settings: Settings, response: str):
        super().__init__(settings)
        self.response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, list[str]]:
        return self.response, []


def build_service(tmp_path, desktop: DesktopActionService | None = None, llm_response: str | None = None, cerebras_api_key: str = ""):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_api_key=cerebras_api_key,
    )
    reports = ReportService(settings)
    memory = MemoryService(settings)
    documents = DocumentService(settings)
    llm = FakeLlmService(settings, llm_response) if llm_response is not None else LlmService(settings)
    study = StudyService(settings, reports, documents, llm)
    service = SafeAgentService(settings, reports, memory, documents, study, desktop or DesktopActionService(), VoiceService(settings))
    return service, memory, documents, study


@pytest.mark.asyncio
async def test_unknown_command_is_blocked_and_audited(tmp_path):
    service, _, _, _ = build_service(tmp_path)

    response = await service.execute(AgentCommandRequest(command_id="run_anything", input_text="run anything"))

    assert response.outcome == "blocked"
    assert response.audit is not None
    assert service.audit_entries()[0].safety_decision == "unknown_command_blocked"


@pytest.mark.asyncio
async def test_invalid_params_are_blocked_before_execution(tmp_path):
    service, _, _, _ = build_service(tmp_path)

    response = await service.execute(AgentCommandRequest(command_id="open_allowlisted_target"))

    assert response.outcome == "blocked"
    assert "target" in response.message.lower()


@pytest.mark.asyncio
async def test_confirm_commands_require_confirmation_then_execute(tmp_path):
    service, memory, _, _ = build_service(tmp_path)
    request = AgentCommandRequest(command_id="save_memory", params={"category": "course", "text": "DBMS exam next week"})

    first = await service.execute(request)
    second = await service.execute(request.model_copy(update={"confirmed": True}))

    assert first.outcome == "confirmation_required"
    assert second.outcome == "success"
    assert memory.list_items()[0].text == "DBMS exam next week"


@pytest.mark.asyncio
async def test_destructive_natural_language_is_blocked(tmp_path):
    service, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("delete my project files")

    assert response is not None
    assert response.outcome == "blocked"
    assert response.audit is not None
    assert response.audit.safety_decision == "blocked_policy"


@pytest.mark.asyncio
async def test_typo_file_explorer_resolves_to_safe_target_and_is_audited(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)

    response = await service.handle_natural_language("open file expolorer")

    assert response is not None
    assert response.outcome == "success"
    assert response.params["target"] == "file_explorer"
    assert desktop.executed == ["open File Explorer"]
    assert response.resolution["source"] == "fuzzy"
    assert response.audit is not None
    assert response.audit.resolution["source"] == "fuzzy"
    assert "File Explorer" in response.message


@pytest.mark.asyncio
async def test_file_explorer_with_local_drive_wins_over_google_drive(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)

    response = await service.handle_natural_language("open file explorer and e drive")

    assert response is not None
    assert response.outcome == "success"
    assert response.params == {"target": "file_explorer", "drive": "E"}
    assert desktop.executed == ["open E: drive"]
    assert "Google Drive" not in response.message


@pytest.mark.asyncio
async def test_open_drive_resolves_and_executes_google_drive_not_google(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)

    for prompt in ("open drive", "open google drive"):
        desktop.executed.clear()
        response = await service.handle_natural_language(prompt)

        assert response is not None
        assert response.outcome == "success"
        assert response.params["target"] == "google_drive"
        assert "Google Drive" in response.message
        assert desktop.executed == ["open Google Drive"]


@pytest.mark.asyncio
async def test_common_safe_target_typos_resolve_without_llm(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)

    cases = [
        ("opne yotube", "youtube"),
        ("open yoootubee", "youtube"),
        ("open yooooutubeeeee", "youtube"),
        ("open yooooootooob", "youtube"),
        ("open gooogle", "google"),
        ("open gmaill", "gmail"),
        ("open calcultor", "calculator"),
        ("open calcullator", "calculator"),
        ("open notpad", "notepad"),
        ("open notepadd", "notepad"),
        ("open fiile expolorer", "file_explorer"),
    ]

    for prompt, target in cases:
        response = await service.handle_natural_language(prompt)
        assert response is not None
        assert response.outcome == "success"
        assert response.params["target"] == target


@pytest.mark.asyncio
async def test_ambiguous_safe_target_typo_requires_confirmation(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)

    response = await service.handle_natural_language("open cal")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.confirmation_required is True
    assert response.resolution["needs_confirmation"] is True
    assert response.resolution["second_best"] >= 0.72
    assert desktop.executed == []


@pytest.mark.asyncio
async def test_confirmed_typo_correction_is_saved_and_reused(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)

    first = await service.handle_natural_language("open cal")
    assert first is not None
    assert first.outcome == "confirmation_required"

    confirmed = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "open cal",
            params=first.params,
            confirmed=True,
            resolution=first.resolution,
        )
    )
    reused = await service.handle_natural_language("open cal")

    assert confirmed.outcome == "success"
    assert confirmed.resolution["correction_saved"] is True
    assert reused is not None
    assert reused.outcome == "success"
    assert reused.resolution["source"] == "correction"


@pytest.mark.asyncio
async def test_medium_confidence_resolution_requires_confirmation(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(tmp_path, desktop=desktop)
    request = AgentCommandRequest(
        command_id="open_allowlisted_target",
        input_text="open filer",
        params={"target": "file_explorer"},
        resolution={
            "source": "fuzzy",
            "confidence": 0.82,
            "matched_alias": "file explorer",
            "normalized_prompt": "open filer",
            "needs_confirmation": True,
        },
    )

    first = await service.execute(request)
    second = await service.execute(request.model_copy(update={"confirmed": True}))

    assert first.outcome == "confirmation_required"
    assert "Did you mean File Explorer" in first.message
    assert desktop.executed == ["open File Explorer"]
    assert second.outcome == "success"


@pytest.mark.asyncio
async def test_llm_fallback_can_only_choose_allowed_registry_command(tmp_path, monkeypatch):
    desktop = RecordingDesktopActionService()
    service, _, _, _ = build_service(
        tmp_path,
        desktop=desktop,
        cerebras_api_key="test-key",
        llm_response='{"command_id":"open_allowlisted_target","params":{"target":"file_explorer"},"confidence":0.94,"matched_alias":"file explorer","reason":"user wants files"}',
    )
    monkeypatch.setattr(service, "_resolve_fuzzy_candidate", lambda *args, **kwargs: None)

    response = await service.handle_natural_language("bring up my local folders")

    assert response is not None
    assert response.outcome == "success"
    assert response.params["target"] == "file_explorer"
    assert response.resolution["source"] == "llm"


@pytest.mark.asyncio
async def test_llm_unknown_command_is_blocked(tmp_path, monkeypatch):
    service, _, _, _ = build_service(
        tmp_path,
        cerebras_api_key="test-key",
        llm_response='{"command_id":"run_anything","params":{},"confidence":0.94,"matched_alias":"anything","reason":"bad candidate"}',
    )
    monkeypatch.setattr(service, "_resolve_fuzzy_candidate", lambda *args, **kwargs: None)

    response = await service.handle_natural_language("do some custom action")

    assert response is not None
    assert response.outcome == "blocked"
    assert response.audit is not None
    assert response.audit.safety_decision == "unknown_command_blocked"


@pytest.mark.asyncio
async def test_llm_invalid_params_are_blocked_by_registry_validation(tmp_path, monkeypatch):
    service, _, _, _ = build_service(
        tmp_path,
        cerebras_api_key="test-key",
        llm_response='{"command_id":"open_allowlisted_target","params":{"target":"registry_editor"},"confidence":0.94,"matched_alias":"registry","reason":"bad target"}',
    )
    monkeypatch.setattr(service, "_resolve_fuzzy_candidate", lambda *args, **kwargs: None)

    response = await service.handle_natural_language("open the registry thing")

    assert response is not None
    assert response.outcome == "blocked"
    assert "not approved" in response.message


@pytest.mark.asyncio
async def test_typo_destructive_request_is_blocked(tmp_path):
    service, _, _, _ = build_service(tmp_path)

    for prompt in ("deleet project files", "opne powershell", "show api ky"):
        response = await service.handle_natural_language(prompt)

        assert response is not None
        assert response.outcome == "blocked"
        assert response.audit is not None
        assert response.audit.safety_decision == "blocked_policy"


@pytest.mark.asyncio
async def test_unknown_app_request_remains_plan_only_for_command_router(tmp_path):
    service, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("open random app")

    assert response is None


@pytest.mark.asyncio
async def test_study_generation_works_in_llm_fallback_mode(tmp_path):
    service, _, _, study = build_service(tmp_path)
    response = await service.execute(
        AgentCommandRequest(
            command_id="generate_study_artifact",
            params={"artifact_type": "flashcards", "topic": "database normalization"},
            confirmed=True,
        )
    )

    assert response.outcome == "success"
    artifacts = study.list_artifacts()
    assert artifacts
    assert "database normalization" in artifacts[0].title.lower()


@pytest.mark.asyncio
async def test_pdf_upload_rejects_non_pdf_and_extracts_text(tmp_path):
    _, _, documents, _ = build_service(tmp_path)

    with pytest.raises(ValueError):
        await documents.upload_pdf(UploadFile(file=BytesIO(b"hello"), filename="note.txt"))

    pdf_bytes = make_text_pdf("Hello Astra PDF")
    record = await documents.upload_pdf(UploadFile(file=BytesIO(pdf_bytes), filename="sample.pdf"))

    assert record.page_count == 1
    assert "Hello Astra" in record.text_preview


def make_text_pdf(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
    content = DecodedStreamObject()
    safe_text = text.replace("(", "\\(").replace(")", "\\)")
    content.set_data(f"BT /F1 24 Tf 100 700 Td ({safe_text}) Tj ET".encode("utf-8"))
    page[NameObject("/Contents")] = content
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
