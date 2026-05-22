import json
from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from app.config import Settings
from app.models import ActionResult, AgentCommandRequest, MockQuestion, MockTestGenerateRequest, MockTestSubmitRequest, Source
from app.services.desktop import DesktopActionService
from app.services.documents import DocumentService
from app.services.llm import LlmService
from app.services.mock_intelligence import MockQuestionQualityValidator
from app.services.memory import MemoryService
from app.services.mock_tests import MockTestService
from app.services.reports import ReportService
from app.services.safe_agent import SafeAgentService
from app.services.study import StudyService
from app.services.voice import VoiceService


_MISSING_LLM_RESPONSE = object()


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
    def __init__(self, settings: Settings, response):
        super().__init__(settings)
        self.response = response
        self.calls: list[tuple[str, str, str | None]] = []

    async def complete(self, system_prompt: str, user_prompt: str, model: str | None = None):
        self.calls.append((system_prompt, user_prompt, model))
        if isinstance(self.response, list):
            if self.response:
                return self.response.pop(0), []
            return None, []
        return self.response, []


class FakeSearchService:
    def __init__(self, sources: list[Source] | None = None):
        self.sources = sources or []
        self.queries: list[str] = []

    async def search_web(self, query: str, max_results: int = 5):
        self.queries.append(query)
        return self.sources[:max_results], []


def build_service(tmp_path, desktop: DesktopActionService | None = None, llm_response=_MISSING_LLM_RESPONSE, cerebras_api_key: str = "", search=None):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_api_key=cerebras_api_key,
    )
    reports = ReportService(settings)
    memory = MemoryService(settings)
    documents = DocumentService(settings)
    llm = FakeLlmService(settings, llm_response) if llm_response is not _MISSING_LLM_RESPONSE else LlmService(settings)
    study = StudyService(settings, reports, documents, llm)
    mock_tests = MockTestService(settings, llm, documents, search)
    service = SafeAgentService(settings, reports, memory, documents, study, mock_tests, desktop or DesktopActionService(), VoiceService(settings))
    return service, memory, documents, study, mock_tests


def mock_question_payload(count: int = 10) -> str:
    prompts = [
        ("Which data structure implements FIFO order in standard queue operations?", ["Queue", "Stack", "Tree", "Graph"], 0, "A queue uses FIFO order, where the first inserted element is removed first.", ["queue", "data structures"]),
        ("What is the time complexity of binary search on a sorted array?", ["O(log n)", "O(n)", "O(n log n)", "O(1)"], 0, "Binary search repeatedly halves the sorted search interval, so it takes O(log n).", ["binary search", "complexity"]),
        ("Which normal form removes transitive dependency in a relation?", ["3NF", "1NF", "2NF", "Unnormalized form"], 0, "Third normal form removes transitive dependency of non-prime attributes on keys.", ["normalization", "dbms"]),
        ("Which OS condition means a resource cannot be forcibly taken from a process?", ["No preemption", "Mutual exclusion", "Hold and wait", "Circular wait"], 0, "No preemption is the deadlock condition where resources are released voluntarily.", ["deadlock", "os"]),
        ("Which protocol provides reliable ordered byte-stream transport?", ["TCP", "UDP", "IP", "ARP"], 0, "TCP uses sequence numbers, acknowledgements, and retransmission for reliable ordered delivery.", ["tcp", "networks"]),
        ("Which automaton model recognizes regular languages?", ["Finite automaton", "Pushdown automaton only", "Turing machine only", "Linear bounded automaton"], 0, "DFA and NFA finite automata recognize the class of regular languages.", ["dfa", "toc"]),
        ("Which cache event occurs when requested data is not present in cache?", ["Cache miss", "Cache hit", "Page replacement", "DMA transfer"], 0, "A cache miss means the processor must fetch the data from a lower memory level.", ["cache", "architecture"]),
        ("Which compiler phase converts character streams into tokens?", ["Lexical analysis", "Code generation", "Register allocation", "Linking"], 0, "Lexical analysis groups input characters into tokens for the parser.", ["compiler", "lexer"]),
        ("Which AI search uses a heuristic estimate to guide path selection?", ["A* search", "Linear search", "Bubble sort", "Round robin"], 0, "A* combines path cost with a heuristic estimate to guide search toward a goal.", ["heuristic", "ai"]),
        ("Which SQL clause filters groups after aggregate functions are computed?", ["HAVING", "WHERE", "FROM", "DISTINCT"], 0, "HAVING is evaluated after grouping and can filter aggregate results.", ["sql", "group by"]),
    ]
    return (
        '{"questions":['
        + ",".join(
            '{"prompt":"%s","options":%s,"correct_option_index":%d,"explanation":"%s","difficulty":"medium","tags":%s}'
            % (
                prompts[(index - 1) % len(prompts)][0],
                str(prompts[(index - 1) % len(prompts)][1]).replace("'", '"'),
                prompts[(index - 1) % len(prompts)][2],
                prompts[(index - 1) % len(prompts)][3],
                str(prompts[(index - 1) % len(prompts)][4]).replace("'", '"'),
            )
            for index in range(1, count + 1)
        )
        + "]}"
    )


@pytest.mark.asyncio
async def test_unknown_command_is_blocked_and_audited(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.execute(AgentCommandRequest(command_id="run_anything", input_text="run anything"))

    assert response.outcome == "blocked"
    assert response.audit is not None
    assert service.audit_entries()[0].safety_decision == "unknown_command_blocked"


@pytest.mark.asyncio
async def test_invalid_params_are_blocked_before_execution(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.execute(AgentCommandRequest(command_id="open_allowlisted_target"))

    assert response.outcome == "blocked"
    assert "target" in response.message.lower()


@pytest.mark.asyncio
async def test_confirm_commands_require_confirmation_then_execute(tmp_path):
    service, memory, _, _, _ = build_service(tmp_path)
    request = AgentCommandRequest(command_id="save_memory", params={"category": "course", "text": "DBMS exam next week"})

    first = await service.execute(request)
    second = await service.execute(request.model_copy(update={"confirmed": True}))

    assert first.outcome == "confirmation_required"
    assert second.outcome == "success"
    assert memory.list_items()[0].text == "DBMS exam next week"


@pytest.mark.asyncio
async def test_destructive_natural_language_is_blocked(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("delete my project files")

    assert response is not None
    assert response.outcome == "blocked"
    assert response.audit is not None
    assert response.audit.safety_decision == "blocked_policy"


@pytest.mark.asyncio
async def test_typo_file_explorer_resolves_to_safe_target_and_is_audited(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)

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
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)

    response = await service.handle_natural_language("open file explorer and e drive")

    assert response is not None
    assert response.outcome == "success"
    assert response.params == {"target": "file_explorer", "drive": "E"}
    assert desktop.executed == ["open E: drive"]
    assert "Google Drive" not in response.message


@pytest.mark.asyncio
async def test_open_drive_resolves_and_executes_google_drive_not_google(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)

    for prompt in ("open drive", "open google drive"):
        desktop.executed.clear()
        response = await service.handle_natural_language(prompt)

        assert response is not None
        assert response.outcome == "success"
        assert response.params["target"] == "google_drive"
        assert "Google Drive" in response.message
        assert desktop.executed == ["open Google Drive"]


@pytest.mark.asyncio
async def test_open_calculator_uses_local_target_when_semantic_llm_omits_params(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _, _ = build_service(
        tmp_path,
        desktop=desktop,
        cerebras_api_key="test-key",
        llm_response='{"command_id":"open_allowlisted_target","intent":"open_allowlisted_target","confidence":0.95,"matched_alias":"calculator","reason":"user wants calculator"}',
    )

    response = await service.handle_natural_language("open calculator")

    assert response is not None
    assert response.outcome == "success"
    assert response.params["target"] == "calculator"
    assert desktop.executed == ["open Calculator"]
    assert service.llm.calls == []


@pytest.mark.asyncio
async def test_common_safe_target_typos_resolve_without_llm(tmp_path):
    desktop = RecordingDesktopActionService()
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)

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
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)

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
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)

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
    service, _, _, _, _ = build_service(tmp_path, desktop=desktop)
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
    service, _, _, _, _ = build_service(
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
    service, _, _, _, _ = build_service(
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
    service, _, _, _, _ = build_service(
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
    service, _, _, _, _ = build_service(tmp_path)

    for prompt in ("deleet project files", "opne powershell", "show api ky"):
        response = await service.handle_natural_language(prompt)

        assert response is not None
        assert response.outcome == "blocked"
        assert response.audit is not None
        assert response.audit.safety_decision == "blocked_policy"


@pytest.mark.asyncio
async def test_unknown_app_request_remains_plan_only_for_command_router(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("open random app")

    assert response is None


@pytest.mark.asyncio
async def test_study_generation_works_in_llm_fallback_mode(tmp_path):
    service, _, _, study, _ = build_service(tmp_path)
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
async def test_mock_test_command_requires_confirmation_then_generates_default_mcq(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)

    first = await service.handle_natural_language("generate mock test on dsa")
    assert first is not None
    assert first.outcome == "confirmation_required"
    assert first.command_id == "generate_mock_test"
    assert first.params["topic"].lower() == "dsa"
    assert first.params["question_count"] == 10

    second = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "generate mock test on dsa",
            params=first.params,
            confirmed=True,
            resolution=first.resolution,
        )
    )

    assert second.outcome == "success"
    assert second.audit is not None
    assert second.audit.safety_decision == "executed_after_validation"
    assert "mock_test" in second.data
    test = mock_tests.list_tests()[0]
    assert test.topic.lower() == "dsa"
    assert test.question_count == 10
    assert test.duration_minutes == 20
    assert test.mode == "mcq"
    assert len(test.questions) == 10
    assert "correct_option_index" not in test.questions[0].model_dump()


@pytest.mark.asyncio
async def test_mock_test_status_complaint_opens_existing_test_without_generation(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)
    existing, _ = await mock_tests.generate(MockTestGenerateRequest(topic="dsa"))

    response = await service.handle_natural_language("dsa mock not created till now")

    assert response is not None
    assert response.outcome == "success"
    assert response.command_id == "open_latest_mock_test"
    assert response.confirmation_required is False
    assert response.data["mock_test"]["id"] == existing.id
    assert len(mock_tests.list_tests()) == 1


@pytest.mark.asyncio
async def test_where_is_my_mock_test_opens_latest_test(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)
    existing, _ = await mock_tests.generate(MockTestGenerateRequest(topic="operating systems"))

    response = await service.handle_natural_language("where is my mock test")

    assert response is not None
    assert response.outcome == "success"
    assert response.command_id == "open_latest_mock_test"
    assert response.data["mock_test"]["id"] == existing.id


@pytest.mark.asyncio
async def test_make_another_mock_test_still_requires_confirmation(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("make another dsa mock test")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.command_id == "generate_mock_test"
    assert response.params["topic"].lower() == "dsa"


@pytest.mark.asyncio
async def test_semantic_parser_understands_ugc_net_cs_pyq_mock_request(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("generate mock test for net ugc cs, only pyq question should be there")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.command_id == "generate_mock_test"
    assert response.params["topic"] == "UGC NET Computer Science"
    assert response.params["source_requirement"] == "pyq_required"
    assert "PYQ required" in response.params["constraints"]
    assert response.resolution["source"] == "semantic_local"


@pytest.mark.asyncio
async def test_semantic_llm_routes_test_me_prompt_to_new_mock_test(tmp_path):
    payload = (
        '{"command_id":"generate_mock_test","intent":"generate_mock_test",'
        '"topic":"Indian History","exam":"","subject":"Current Affairs",'
        '"question_count":10,"difficulty":"mixed","mode":"mcq","duration_minutes":20,'
        '"requires_sources":false,"source_requirement":"none","source_mode":"uploaded_docs",'
        '"source_query":"Indian History current affairs","constraints":["current affairs context"],'
        '"is_new_creation":true,"is_existing_item_request":false,'
        '"confidence":0.96,"matched_alias":"test me on current affairs","reason":"The user wants a new assessment on Indian history."}'
    )
    service, _, _, _, _ = build_service(tmp_path, llm_response=payload, cerebras_api_key="test-key")

    response = await service.handle_natural_language("test me on current affairs on the topic indian history")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.command_id == "generate_mock_test"
    assert response.params["topic"] == "Indian History"
    assert response.params["subject"] == "Current Affairs"
    assert response.resolution["source"] == "llm"


@pytest.mark.asyncio
async def test_semantic_local_routes_test_me_prompt_without_opening_latest(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)
    existing, _ = await mock_tests.generate(MockTestGenerateRequest(topic="discrete mathematics"))

    response = await service.handle_natural_language("test me on current affairs on the topic indian history")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.command_id == "generate_mock_test"
    assert response.params["topic"].lower() == "indian history"
    assert response.params["subject"] == "Current Affairs"
    assert response.data == {}
    assert mock_tests.list_tests()[0].id == existing.id


@pytest.mark.asyncio
async def test_semantic_llm_new_creation_flag_overrides_wrong_open_latest_choice(tmp_path):
    payload = (
        '{"command_id":"open_latest_mock_test","intent":"generate_mock_test",'
        '"topic":"Indian History","exam":"","subject":"Current Affairs",'
        '"question_count":10,"difficulty":"mixed","mode":"mcq","duration_minutes":20,'
        '"requires_sources":false,"source_requirement":"none","source_mode":"uploaded_docs",'
        '"source_query":"Indian History","constraints":[],'
        '"is_new_creation":true,"is_existing_item_request":false,'
        '"confidence":0.93,"matched_alias":"test me","reason":"The wording asks for a new assessment."}'
    )
    service, _, _, _, _ = build_service(tmp_path, llm_response=payload, cerebras_api_key="test-key")

    response = await service.handle_natural_language("test me on current affairs on the topic indian history")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.command_id == "generate_mock_test"
    assert response.params["topic"] == "Indian History"
    assert response.resolution["semantic_override"] == "open_latest_to_generate"


@pytest.mark.asyncio
async def test_pyq_generation_without_sources_does_not_use_generic_fallback(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)

    first = await service.handle_natural_language("generate mock test for net ugc cs only pyq questions")
    assert first is not None

    second = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "generate mock test for net ugc cs only pyq questions",
            params=first.params,
            confirmed=True,
            resolution=first.resolution,
        )
    )

    assert second.outcome == "failure"
    assert "source material" in second.message.lower()
    assert "Use uploaded PDF" in second.data["source_actions"]
    assert "Generate PYQ-style practice" in second.data["source_actions"]
    assert mock_tests.list_tests() == []


@pytest.mark.asyncio
async def test_source_backed_pyq_generation_stores_sources_and_metadata(tmp_path):
    source = Source(
        title="Uploaded UGC NET Computer Science PYQ Set",
        url="document:test-pyq",
        snippet="Uploaded source text with actual UGC NET Computer Science questions.",
        provider="Uploaded PDF",
    )
    _, _, _, _, mock_tests = build_service(tmp_path, cerebras_api_key="test-key", llm_response=mock_question_payload(10))

    test, _ = await mock_tests.generate(
        MockTestGenerateRequest(
            topic="UGC NET Computer Science",
            source_requirement="pyq_required",
            source_mode="uploaded_docs",
            source_text=(
                "UGC NET Computer Science previous year questions. "
                "1. Which data structure uses FIFO order? (A) Stack (B) Queue (C) Tree (D) Graph Answer: B. "
                "2. What is the time complexity of binary search? (A) O(n) (B) O(log n) (C) O(n log n) (D) O(1) Answer: B. "
                "3. Which normal form removes transitive dependency? (A) 1NF (B) 2NF (C) 3NF (D) BCNF Answer: C."
            ),
            source_refs=[source],
        )
    )

    assert test.topic == "UGC NET Computer Science"
    assert test.source_requirement == "pyq_required"
    assert test.source == "pyq_required"
    assert test.sources[0].provider == "Uploaded PDF"
    assert test.questions[0].source_refs


@pytest.mark.asyncio
async def test_pyq_generation_never_queries_web_search(tmp_path):
    search = FakeSearchService(
        [
            Source(
                title="Thin UGC NET PYQ listing",
                url="https://example.com/thin",
                snippet="UGC NET CS PYQ downloads and preparation links.",
                provider="TestSearch",
            )
        ]
    )
    service, _, _, _, mock_tests = build_service(tmp_path, cerebras_api_key="test-key", llm_response=mock_question_payload(10), search=search)

    first = await service.handle_natural_language("generate mock test for net ugc cs only pyq questions")
    assert first is not None
    second = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "generate mock test for net ugc cs only pyq questions",
            params={**first.params, "source_mode": "uploaded_docs"},
            confirmed=True,
            resolution=first.resolution,
        )
    )

    assert second.outcome == "failure"
    assert "source material" in second.message.lower()
    assert "Search web sources" not in second.data["source_actions"]
    assert "Use uploaded PDF" in second.data["source_actions"]
    assert search.queries == []
    assert mock_tests.list_tests() == []


@pytest.mark.asyncio
async def test_vague_ugc_net_pyq_clarifies_instead_of_executing(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)

    response = await service.handle_natural_language("ugc net pyq")

    assert response is not None
    assert response.outcome == "success"
    assert response.command_id == "clarify_agent_intent"
    assert "need source material" in response.message.lower()
    assert mock_tests.list_tests() == []


@pytest.mark.asyncio
async def test_agent_greeting_is_not_misrouted_to_pyq_clarification(tmp_path):
    service, _, _, _, _ = build_service(tmp_path)

    response = await service.handle_natural_language("hiii")

    assert response is None


@pytest.mark.asyncio
async def test_llm_clarification_is_ignored_when_prompt_does_not_mention_pyq(tmp_path):
    llm_payload = (
        '{"command_id":"clarify_agent_intent","intent":"clarify","topic":"hiii","needs_clarification":true,'
        '"confidence":0.94,"matched_alias":"clarify","reason":"needs clarification"}'
    )
    service, _, _, _, _ = build_service(tmp_path, cerebras_api_key="test-key", llm_response=llm_payload)

    response = await service.handle_natural_language("hiii")

    assert response is None


@pytest.mark.asyncio
async def test_pyq_style_ugc_net_generates_cs_questions_without_fake_pyq_sources(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)

    first = await service.handle_natural_language("generate mock test for net ugc cs pyq style")
    assert first is not None
    assert first.outcome == "confirmation_required"
    assert first.params["topic"] == "UGC NET Computer Science"
    assert first.params["source_requirement"] == "none"
    assert "PYQ-style practice" in first.params["constraints"]

    second = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "generate mock test for net ugc cs pyq style",
            params=first.params,
            confirmed=True,
            resolution=first.resolution,
        )
    )

    assert second.outcome == "success"
    test = mock_tests.list_tests()[0]
    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert test.generation_mode == "pyq_style"
    assert test.source_requirement == "none"
    assert "core purpose" not in prompts
    assert any(term in prompts for term in ["normal form", "binary search", "deadlock", "tcp", "automaton"])


@pytest.mark.asyncio
async def test_generic_ugc_net_pyq_style_uses_paper_one_profile_without_cerebras(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)

    first = await service.handle_natural_language("generate mock test for ugc net pyq based only")
    assert first is not None
    assert first.outcome == "confirmation_required"
    assert first.params["topic"] == "UGC NET"
    assert first.params["exam"] == "UGC NET"
    assert first.params["subject"] == "Paper I: Teaching and Research Aptitude"
    assert first.params["source_requirement"] == "pyq_required"

    pyq_style = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "generate mock test for ugc net pyq based only",
            params={**first.params, "source_requirement": "none", "source_mode": "uploaded_docs", "constraints": ["PYQ-style practice"]},
            confirmed=True,
            resolution=first.resolution,
        )
    )

    assert pyq_style.outcome == "success"
    test = mock_tests.list_tests()[0]
    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert test.exam == "UGC NET"
    assert test.subject == "Paper I: Teaching and Research Aptitude"
    assert test.generation_mode == "pyq_style"
    assert any(term in prompts for term in ["teaching", "research", "communication", "syllogism", "hypothesis"])
    assert "core purpose" not in prompts
    assert "data structure" not in prompts


@pytest.mark.asyncio
async def test_gate_cse_dbms_uses_database_blueprint(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)

    test, _ = await mock_tests.generate(MockTestGenerateRequest(topic="GATE CSE DBMS"))

    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert test.exam == "GATE"
    assert test.subject == "Computer Science"
    assert test.generation_mode == "profile_based"
    assert any(term in prompts for term in ["normal form", "sql", "transaction", "b+ tree"])
    assert "core purpose" not in prompts


@pytest.mark.asyncio
async def test_neet_biology_genetics_uses_biology_blueprint(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)

    test, _ = await mock_tests.generate(MockTestGenerateRequest(topic="NEET biology genetics"))

    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert test.exam == "NEET"
    assert test.subject == "Biology"
    assert test.generation_mode == "profile_based"
    assert any(term in prompts for term in ["monohybrid", "codon", "dna", "allele"])
    assert "prioritize first" not in prompts


@pytest.mark.asyncio
async def test_unknown_exam_uses_llm_blueprint_without_web_search(tmp_path):
    planner_payload = (
        '{"exam":"Example Board","subject":"Robotics","syllabus_units":["Sensors","Actuators","PID Control"],'
        '"question_style":"Robotics MCQs with control-system reasoning.",'
        '"expected_terms":["sensor","actuator","pid","controller"],"confidence":0.81}'
    )
    question_payload = (
        '{"questions":['
        + ",".join(
            '{"prompt":"In a PID controller, which term responds to accumulated steady-state error %d?",'
            '"options":["Integral term","Derivative term","Sensor noise","Actuator saturation"],'
            '"correct_option_index":0,"explanation":"The integral term accumulates error over time and helps remove steady-state offset.",'
            '"difficulty":"medium","tags":["pid","controller"]}'
            % index
            for index in range(1, 11)
        )
        + "]}"
    )
    repair_payload = (
        '{"questions":['
        + ",".join(
            '{"prompt":"In a PID controller, which term responds to accumulated steady-state error repair %d?",'
            '"options":["Integral term","Derivative term","Sensor noise","Actuator saturation"],'
            '"correct_option_index":0,"explanation":"The integral term accumulates error over time and helps remove steady-state offset.",'
            '"difficulty":"medium","tags":["pid","controller"]}'
            % index
            for index in range(11, 15)
        )
        + "]}"
    )
    search = FakeSearchService([])
    _, _, _, _, mock_tests = build_service(tmp_path, llm_response=[planner_payload, question_payload, repair_payload], cerebras_api_key="test-key", search=search)

    test, setup = await mock_tests.generate(MockTestGenerateRequest(topic="Example Board robotics"))

    assert setup == []
    assert search.queries == []
    assert test.generation_mode == "llm_planned"
    assert test.blueprint_source == "llm_planner"
    assert test.exam == "Example Board"
    assert "PID Control" in test.syllabus_units


@pytest.mark.asyncio
async def test_single_question_rescue_completes_unknown_topic_after_bad_batches(tmp_path):
    planner_payload = (
        '{"exam":"Example Board","subject":"Robotics","syllabus_units":["Sensors","Actuators","PID Control"],'
        '"question_style":"Robotics MCQs with control-system reasoning.",'
        '"expected_terms":["sensor","actuator","pid","controller"],"confidence":0.81}'
    )
    bad_batches = ["not valid json"] * 5
    rescue_payloads = [
        json.dumps(
            {
                "questions": [
                    {
                        "prompt": f"A robot controller reads sensor value {index} and target 10. Which PID term reacts to accumulated past error?",
                        "options": ["Integral term", "Derivative term", "Sensor casing", "Actuator voltage only"],
                        "correct_option_index": 0,
                        "explanation": "The integral term sums controller error over time and reduces steady-state offset.",
                        "difficulty": "medium",
                        "tags": ["pid", "controller", "sensor"],
                    }
                ]
            }
        )
        for index in range(1, 4)
    ]
    _, _, _, _, mock_tests = build_service(
        tmp_path,
        llm_response=[planner_payload, *bad_batches, *rescue_payloads],
        cerebras_api_key="test-key",
    )

    test, setup = await mock_tests.generate(MockTestGenerateRequest(topic="Example Board robotics", question_count=3))

    assert setup == []
    assert test.question_count == 3
    assert test.quality_score == 1.0
    assert all("robot controller" in question.prompt.lower() for question in test.questions)


@pytest.mark.asyncio
async def test_unknown_javascript_topic_uses_llm_blueprint_and_repair_loop(tmp_path):
    planner_payload = (
        '{"exam":"","subject":"JavaScript Programming",'
        '"syllabus_units":["Variables and Scope","Closures","Promises and Async Await","Event Loop","Prototypes","Arrays and Objects"],'
        '"question_style":"JavaScript coding MCQs with code output and runtime behavior.",'
        '"expected_terms":["javascript","closure","promise","async","event loop","prototype","this","array","object","hoisting"],'
        '"confidence":0.88}'
    )
    first_batch = javascript_payload(
        [
            generic_question("JavaScript", index) for index in range(1, 6)
        ]
        + [
            js_question(index, f"candidate {index}") for index in range(6, 13)
        ]
    )
    repair_batch = javascript_payload([js_question(index, f"repair {index}") for index in range(13, 21)])
    _, _, _, _, mock_tests = build_service(
        tmp_path,
        llm_response=[planner_payload, first_batch, repair_batch],
        cerebras_api_key="test-key",
    )

    test, setup = await mock_tests.generate(MockTestGenerateRequest(topic="JavaScript", question_count=10))

    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert setup == []
    assert test.generation_mode == "llm_planned"
    assert test.blueprint_source == "llm_planner"
    assert test.subject == "JavaScript Programming"
    assert len(test.questions) == 10
    assert "core purpose" not in prompts
    assert "candidate 6" in prompts
    assert "repair" in prompts
    assert len(mock_tests.llm.calls) == 3  # type: ignore[attr-defined]
    assert "rejected_reasons" in mock_tests.llm.calls[-1][1]  # type: ignore[attr-defined]
    assert "candidate_count" in mock_tests.llm.calls[1][1]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_unknown_discrete_math_numerical_uses_llm_batches_and_constraints(tmp_path):
    planner_payload = (
        '{"exam":"","subject":"Discrete Mathematics",'
        '"syllabus_units":["Counting Principles","Set Theory","Graph Theory","Recurrence Relations"],'
        '"question_style":"Numerical discrete mathematics MCQs with calculations.",'
        '"expected_terms":["permutation","combination","subset","graph degree","recurrence","probability"],'
        '"confidence":0.9}'
    )
    first_batch = json.dumps(
        {
            "questions": [
                {
                    "prompt": "Given P is true and Q is false, what is the truth value of P implies Q?",
                    "options": ["True", "False", "Both true and false", "Cannot be determined"],
                    "correct_option_index": 1,
                    "explanation": "This is a non-numerical truth-table item and should be filtered.",
                    "difficulty": "easy",
                    "tags": ["logic"],
                },
                *[discrete_numerical_question(index, "batch") for index in range(1, 10)],
            ]
        }
    )
    repair_batch = json.dumps({"questions": [discrete_numerical_question(index, "repair") for index in range(10, 13)]})
    _, _, _, _, mock_tests = build_service(
        tmp_path,
        llm_response=[planner_payload, first_batch, repair_batch],
        cerebras_api_key="test-key",
    )

    test, setup = await mock_tests.generate(
        MockTestGenerateRequest(topic="Example Board Combinatorics", constraints=["only numerical questions"], question_count=10)
    )

    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert setup == []
    assert test.generation_mode == "llm_planned"
    assert test.subject == "Discrete Mathematics"
    assert len(test.questions) == 10
    assert "truth value" not in prompts
    assert all(any(char.isdigit() for char in " ".join([question.prompt, *question.options])) for question in test.questions)
    first_generation_prompt = json.loads(mock_tests.llm.calls[1][1])  # type: ignore[attr-defined]
    assert 10 <= first_generation_prompt["candidate_count"] <= 16
    assert first_generation_prompt["constraints"] == ["only numerical questions"]
    assert any("numerical-only" in rule for rule in first_generation_prompt["quality_rules"])


def test_questions_from_llm_salvages_complete_questions_from_truncated_json(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)
    request = MockTestGenerateRequest(topic="Discrete Mathematics", constraints=["only numerical questions"], question_count=2)
    complete_one = json.dumps(discrete_numerical_question(1, "partial"))
    complete_two = json.dumps(discrete_numerical_question(2, "partial"))
    raw = f'```json\n{{"questions":[{complete_one},{complete_two},{{"prompt":"unfinished"'

    questions = mock_tests._questions_from_llm(raw, request, max_questions=4)

    assert len(questions) == 2
    assert questions[0].prompt.startswith("How many subsets")


def test_computer_science_discrete_math_matches_discrete_profile_not_school_science(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)
    request = mock_tests._clean_request(
        MockTestGenerateRequest(topic="Computer Science Discrete Mathematics", subject="Computer Science")
    )

    profile = mock_tests.intelligence._match_profile(request)

    assert profile is not None
    assert profile.id == "discrete_math"


@pytest.mark.asyncio
async def test_discrete_math_profile_generates_numerical_questions_without_llm(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)

    test, setup = await mock_tests.generate(
        MockTestGenerateRequest(topic="discrete mathematics", constraints=["only numerical questions"], question_count=10)
    )

    assert setup == []
    assert test.subject == "Discrete Mathematics"
    assert test.generation_mode == "profile_based"
    assert test.quality_score == 1.0
    assert all(any(char.isdigit() for char in " ".join([question.prompt, *question.options])) for question in test.questions)
    assert "mitochondria" not in " ".join(question.prompt.lower() for question in test.questions)


@pytest.mark.asyncio
async def test_cyber_security_profile_generates_hard_numerical_questions_without_llm(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)

    test, setup = await mock_tests.generate(
        MockTestGenerateRequest(
            topic="cyber security numerical questions",
            constraints=["numerical questions", "toughest exam difficulty"],
            difficulty="hard",
            question_count=10,
        )
    )

    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert setup == []
    assert test.subject == "Cyber Security"
    assert test.generation_mode == "profile_based"
    assert test.quality_score == 1.0
    assert all(question.difficulty == "hard" for question in test.questions)
    assert all(any(char.isdigit() for char in " ".join([question.prompt, *question.options])) for question in test.questions)
    assert "rsa" in prompts or "risk" in prompts


@pytest.mark.asyncio
async def test_llm_blueprint_uses_syllabus_terms_for_quality_matching(tmp_path):
    planner_payload = (
        '{"exam":"","subject":"Discrete Mathematics",'
        '"syllabus_units":["Combinatorics"],'
        '"question_style":"Numerical combinatorics MCQs.",'
        '"expected_terms":["permutation"],'
        '"confidence":0.9}'
    )
    question_payload = json.dumps(
        {
            "questions": [
                {
                    "prompt": f"How many ways can {index + 3} distinct books be arranged on a shelf?",
                    "options": [
                        str(math_factorial(index + 3)),
                        str(math_factorial(index + 3) + 1),
                        str(math_factorial(index + 3) + index + 3),
                        str(math_factorial(index + 2)),
                    ],
                    "correct_option_index": 0,
                    "explanation": f"Arranging {index + 3} distinct items gives factorial count {(index + 3)}!, so the listed value is correct.",
                    "difficulty": ["easy", "medium", "hard"][index % 3],
                    "tags": ["Combinatorics"],
                }
                for index in range(10)
            ]
        }
    )
    repair_payload = json.dumps(
        {
            "questions": [
                {
                    "prompt": f"How many ways can {index + 3} distinct books be arranged on a shelf?",
                    "options": [
                        str(math_factorial(index + 3)),
                        str(math_factorial(index + 3) + 1),
                        str(math_factorial(index + 3) + index + 3),
                        str(math_factorial(index + 2)),
                    ],
                    "correct_option_index": 0,
                    "explanation": f"Arranging {index + 3} distinct items gives factorial count {(index + 3)}!, so the listed value is correct.",
                    "difficulty": ["easy", "medium", "hard"][index % 3],
                    "tags": ["Combinatorics"],
                }
                for index in range(10, 14)
            ]
        }
    )
    _, _, _, _, mock_tests = build_service(
        tmp_path,
        llm_response=[planner_payload, question_payload, repair_payload],
        cerebras_api_key="test-key",
    )

    test, _ = await mock_tests.generate(MockTestGenerateRequest(topic="Example Board Combinatorics", question_count=10))

    assert test.generation_mode == "llm_planned"
    assert test.quality_score == 1.0
    assert "Combinatorics" in test.syllabus_units


@pytest.mark.asyncio
async def test_unknown_topic_does_not_save_generic_llm_filler_after_repairs(tmp_path):
    planner_payload = (
        '{"exam":"","subject":"JavaScript Programming",'
        '"syllabus_units":["Variables and Scope","Closures"],'
        '"question_style":"JavaScript coding MCQs.",'
        '"expected_terms":["javascript","closure","scope"],"confidence":0.8}'
    )
    bad_payload = javascript_payload([generic_question("JavaScript", index) for index in range(1, 16)])
    _, _, _, _, mock_tests = build_service(
        tmp_path,
        llm_response=[planner_payload, bad_payload, bad_payload, bad_payload],
        cerebras_api_key="test-key",
    )

    with pytest.raises(ValueError, match="without falling back to generic filler"):
        await mock_tests.generate(MockTestGenerateRequest(topic="JavaScript", question_count=10))

    assert mock_tests.list_tests() == []


def javascript_payload(items: list[dict]) -> str:
    import json

    return json.dumps({"questions": items})


def generic_question(topic: str, index: int) -> dict:
    return {
        "prompt": f"Which option best describes the core purpose of {topic} practice question {index}?",
        "options": ["Clear fundamentals", "Random guessing", "Skipping examples", "Ignoring constraints"],
        "correct_option_index": 0,
        "explanation": f"The correct answer is a generic study habit for {topic}.",
        "difficulty": "medium",
        "tags": ["fallback"],
    }


def js_question(index: int, label: str) -> dict:
    difficulty = ["easy", "medium", "hard"][index % 3]
    return {
        "prompt": (
            f"Consider JavaScript {label}: what is printed by this code?\n\n"
            "const values = [1, 2, 3];\n"
            "const doubled = values.map((value) => value * 2);\n"
            "console.log(doubled.length);"
        ),
        "options": ["3", "6", "undefined", "A TypeError is thrown"],
        "correct_option_index": 0,
        "explanation": "JavaScript Array.map returns a new array with the same length, so the printed length is 3.",
        "difficulty": difficulty,
        "tags": ["javascript", "array", "map"],
    }


def discrete_numerical_question(index: int, label: str) -> dict:
    n = index + 2
    correct = 2**n
    difficulty = ["easy", "medium", "hard"][index % 3]
    return {
        "prompt": f"How many subsets does a set with {n} elements have in discrete mathematics {label} {index}?",
        "options": [str(correct), str(correct - 1), str(correct + n), str(n * 2)],
        "correct_option_index": 0,
        "explanation": f"A set with n elements has 2^n subsets; for n={n}, 2^{n}={correct}.",
        "difficulty": difficulty,
        "tags": ["subset", "combination", "discrete mathematics"],
    }


def math_factorial(value: int) -> int:
    result = 1
    for number in range(2, value + 1):
        result *= number
    return result


def test_quality_gate_rejects_generic_study_advice_questions():
    validator = MockQuestionQualityValidator()
    weak = MockQuestion(
        id="q1",
        prompt="Which option best describes the core purpose of UGC NET Computer Science?",
        options=["Solving problems with structured concepts", "Only memorizing definitions", "Avoiding practice problems", "Ignoring constraints"],
        correct_option_index=0,
        explanation="The correct answer focuses on practical understanding and review for UGC NET Computer Science.",
        difficulty="medium",
        tags=["fallback"],
    )

    accepted, score, warnings = validator.validate(
        [weak],
        mock_tests_blueprint_for_quality(),
        required_count=1,
        strict_terms=True,
    )
    assert accepted == []
    assert score == 0.0
    assert any("generic" in item for item in warnings)


def mock_tests_blueprint_for_quality():
    from app.services.mock_intelligence import MockBlueprint

    return MockBlueprint(
        topic="UGC NET Computer Science",
        generation_mode="profile_based",
        blueprint_source="profile:ugc_net_cs",
        syllabus_units=["Data Structures and Algorithms", "Database Management Systems"],
        expected_terms=["algorithm", "normalization", "transaction", "deadlock"],
    )


def mock_tests_blueprint_for_python_quality():
    from app.services.mock_intelligence import MockBlueprint

    return MockBlueprint(
        topic="python",
        subject="Python Programming",
        generation_mode="profile_based",
        blueprint_source="profile:python_programming",
        syllabus_units=["Python Syntax and Data Types", "Functions and Scope"],
        expected_terms=["python", "generator", "function", "yield"],
    )


@pytest.mark.asyncio
async def test_mock_test_generation_uses_profile_when_llm_returns_empty_payload(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path, llm_response=None, cerebras_api_key="test-key")

    test, setup = await mock_tests.generate(MockTestGenerateRequest(topic="dsa"))

    assert test.topic == "dsa"
    assert test.question_count == 10
    assert len(test.questions) == 10
    assert test.source == "profile_based"
    assert test.generation_mode == "profile_based"
    assert test.quality_score == 1.0
    assert setup == []


@pytest.mark.asyncio
async def test_create_python_test_generates_python_profile_mock(tmp_path):
    service, _, _, _, mock_tests = build_service(tmp_path)

    first = await service.handle_natural_language("create python test")
    assert first is not None
    assert first.outcome == "confirmation_required"
    assert first.command_id == "generate_mock_test"
    assert first.params["topic"] == "python"

    second = await service.execute(
        AgentCommandRequest(
            command_id=first.command_id,
            input_text=first.audit.input_text if first.audit else "create python test",
            params=first.params,
            confirmed=True,
            resolution=first.resolution,
        )
    )

    assert second.outcome == "success"
    test = mock_tests.list_tests()[0]
    prompts = " ".join(question.prompt.lower() for question in test.questions)
    assert test.topic == "python"
    assert test.subject == "Python Programming"
    assert test.generation_mode == "profile_based"
    assert len(test.questions) == 10
    assert any(term in prompts for term in ["list", "dictionary", "exception", "decorator", "lambda"])
    assert "python python" not in second.message.lower()
    assert "core purpose" not in prompts


@pytest.mark.asyncio
async def test_profile_mock_respects_selected_easy_and_hard_difficulty(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)

    easy, _ = await mock_tests.generate(MockTestGenerateRequest(topic="python", difficulty="easy"))
    hard, _ = await mock_tests.generate(MockTestGenerateRequest(topic="python", difficulty="hard"))

    assert easy.difficulty == "easy"
    assert hard.difficulty == "hard"
    assert {question.difficulty for question in easy.questions} == {"easy"}
    assert {question.difficulty for question in hard.questions} == {"hard"}

    easy_prompts = " ".join(question.prompt.lower() for question in easy.questions)
    hard_prompts = " ".join(question.prompt.lower() for question in hard.questions)
    assert any(marker in easy_prompts for marker in ["direct concept check", "which", "what"])
    assert any(marker in hard_prompts for marker in ["consider", "trace", "output", "edge case", "given"])
    assert easy.questions[0].prompt != hard.questions[0].prompt


@pytest.mark.asyncio
async def test_mixed_mock_uses_real_difficulty_distribution(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)

    test, _ = await mock_tests.generate(MockTestGenerateRequest(topic="dsa", difficulty="mixed"))

    difficulties = [question.difficulty for question in test.questions]
    assert test.difficulty == "mixed"
    assert {"easy", "medium", "hard"}.issubset(set(difficulties))
    assert difficulties[:4] == ["easy", "medium", "medium", "hard"]


def test_quality_gate_rejects_direct_recall_when_hard_requested():
    validator = MockQuestionQualityValidator()
    direct = MockQuestion(
        id="q1",
        prompt="Which keyword creates a Python generator?",
        options=["yield", "lambda", "class", "import"],
        correct_option_index=0,
        explanation="The yield keyword makes a Python function produce a generator.",
        difficulty="hard",
        tags=["python", "generator"],
    )

    accepted, score, warnings = validator.validate(
        [direct],
        mock_tests_blueprint_for_python_quality(),
        required_count=1,
        strict_terms=True,
        requested_difficulty="hard",
    )

    assert accepted == []
    assert score == 0.0
    assert any("too direct" in item for item in warnings)


@pytest.mark.asyncio
async def test_duplicate_python_topic_parts_from_llm_are_deduped(tmp_path):
    payload = (
        '{"command_id":"generate_mock_test","intent":"generate_mock_test","topic":"python python",'
        '"exam":"","subject":"","question_count":10,"difficulty":"mixed","mode":"mcq",'
        '"duration_minutes":20,"requires_sources":false,"source_requirement":"none",'
        '"source_mode":"uploaded_docs","constraints":[],"confidence":0.95,'
        '"matched_alias":"create python test","reason":"user wants a Python mock test"}'
    )
    service, _, _, _, _ = build_service(tmp_path, llm_response=payload, cerebras_api_key="test-key")

    response = await service.handle_natural_language("create python test")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert response.params["topic"] == "python"
    assert response.params["topic"] != "python python"


@pytest.mark.asyncio
async def test_topic_dedupe_preserves_specific_focus_terms(tmp_path):
    payload = (
        '{"command_id":"generate_mock_test","intent":"generate_mock_test","topic":"GATE CSE DBMS",'
        '"exam":"GATE","subject":"Computer Science","question_count":10,"difficulty":"mixed","mode":"mcq",'
        '"duration_minutes":20,"requires_sources":false,"source_requirement":"none",'
        '"source_mode":"uploaded_docs","constraints":[],"confidence":0.95,'
        '"matched_alias":"create gate cse dbms test","reason":"user wants a GATE CSE DBMS test"}'
    )
    service, _, _, _, _ = build_service(tmp_path, llm_response=payload, cerebras_api_key="test-key")

    response = await service.handle_natural_language("create gate cse dbms test")

    assert response is not None
    assert response.outcome == "confirmation_required"
    assert "DBMS" in response.params["topic"]
    assert response.params["exam"] == "GATE"
    assert response.params["subject"] == "Computer Science"


@pytest.mark.asyncio
async def test_mock_test_start_and_submit_scores_answers(tmp_path):
    _, _, _, _, mock_tests = build_service(tmp_path)
    test, _ = await mock_tests.generate(MockTestGenerateRequest(topic="dsa"))
    _, attempt = mock_tests.start_attempt(test.id)
    answers = {question.id: 0 for question in test.questions}

    result = mock_tests.submit_attempt(test.id, attempt.id, MockTestSubmitRequest(answers=answers, elapsed_seconds=90))

    assert result.score == result.total
    assert result.percentage == 100
    assert result.correct_count == 10
    assert result.elapsed_seconds == 90
    assert result.attempt.status == "submitted"
    assert result.review[0].selected_option_index == 0
    assert result.review[0].correct_option_index == 0
    assert result.review[0].explanation


@pytest.mark.asyncio
async def test_pdf_upload_rejects_non_pdf_and_extracts_text(tmp_path):
    _, _, documents, _, _ = build_service(tmp_path)

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
