import pytest

from app.config import Settings
from app.models import ChatResponse, CommandRequest, ResearchRequest
from app.services.agents import AstraAgentSystem
from app.services.commands import CommandService
from app.services.desktop import DesktopActionService


class RecordingLlm:
    def __init__(self, answer: str = "ok"):
        self.answer = answer
        self.models: list[str | None] = []
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    async def complete(self, system_prompt: str, user_prompt: str, model: str | None = None):
        self.models.append(model)
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        return self.answer, []


class EmptySearch:
    async def search_web(self, query: str, max_results: int = 5):
        return [], []

    async def search_academic(self, query: str, max_results: int = 5):
        return [], []


class EmptySafeAgent:
    async def handle_natural_language(self, text: str, confirmed: bool = False):
        return None


def build_command_service(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_fast_model="fast-model",
        cerebras_pro_model="pro-model",
    )
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm()
    agent_system.llm = recorder  # type: ignore[assignment]
    service = CommandService(agent_system, DesktopActionService(), object(), EmptySafeAgent())  # type: ignore[arg-type]
    return service, agent_system, recorder


@pytest.mark.asyncio
async def test_cockpit_command_uses_fast_model_by_default(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    await service.handle(CommandRequest(text="summarize my DBMS notes", mode="cockpit", astra_pro=False))

    assert recorder.models == ["fast-model"]


@pytest.mark.asyncio
async def test_cockpit_command_uses_pro_model_when_astra_pro_enabled(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    await service.handle(CommandRequest(text="summarize my DBMS notes", mode="cockpit", astra_pro=True))

    assert recorder.models == ["pro-model"]


@pytest.mark.asyncio
async def test_agent_mode_forces_pro_model(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    await service.handle(CommandRequest(text="summarize my DBMS notes", mode="agents", astra_pro=False))

    assert recorder.models == ["pro-model"]


@pytest.mark.asyncio
async def test_agent_mode_greeting_is_plain_chat_without_intent_plan(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    response = await service.handle(CommandRequest(text="hiii", mode="agents", astra_pro=False))

    assert response.intent == "chat"
    assert response.display_text == "Hello! I'm Astra. How can I help?"
    assert "Intent:" not in response.display_text
    assert "Plan:" not in response.display_text
    assert recorder.models == []


@pytest.mark.asyncio
async def test_agent_mode_identity_is_plain_chat_without_intent_plan(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    response = await service.handle(CommandRequest(text="what is your name", mode="agents", astra_pro=False))

    assert response.intent == "chat"
    assert response.display_text == "My name is Astra. I'm your voice-first AI assistant."
    assert "Intent:" not in response.display_text
    assert "Plan:" not in response.display_text
    assert recorder.models == []


@pytest.mark.asyncio
async def test_research_uses_pro_model(tmp_path):
    _, agent_system, recorder = build_command_service(tmp_path)
    agent_system.search = EmptySearch()  # type: ignore[assignment]

    await agent_system.research(ResearchRequest(topic="multi agent study assistant", source_mode="web"))

    assert recorder.models == ["pro-model"]


@pytest.mark.asyncio
async def test_direct_chat_request_can_select_fast_or_pro(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_fast_model="fast-model",
        cerebras_pro_model="pro-model",
    )
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm()
    agent_system.llm = recorder  # type: ignore[assignment]

    fast: ChatResponse = await agent_system.chat("hello", "cockpit", astra_pro=False)
    pro: ChatResponse = await agent_system.chat("hello", "cockpit", astra_pro=True)

    assert fast.answer == "ok"
    assert pro.answer == "ok"
    assert recorder.models == ["fast-model", "pro-model"]


@pytest.mark.asyncio
async def test_agent_chat_prompt_and_cleanup_hide_internal_sections(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_fast_model="fast-model",
        cerebras_pro_model="pro-model",
    )
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm(
        "**Intent:** User requests identification.\n"
        "**Plan:** State name and status.\n"
        "**Execution:** My name is Astra. I am ready to help."
    )
    agent_system.llm = recorder  # type: ignore[assignment]

    response = await agent_system.chat("what is your name", "agents", astra_pro=False)

    assert response.answer == "My name is Astra. I am ready to help."
    assert "Intent:" not in response.answer
    assert "Plan:" not in response.answer
    assert "Do not show internal labels" in recorder.system_prompts[0]
    assert "identify intent" not in recorder.system_prompts[0]
    assert recorder.models == ["pro-model"]
