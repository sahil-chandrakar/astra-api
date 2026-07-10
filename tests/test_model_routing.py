import asyncio

import pytest

from app.config import Settings
from app.models import AgentCommandResponse, AutomationRun, ChatResponse, CommandRequest, LlmProfileConfig, LlmSettingsUpdateRequest, ResearchRequest
from app.services.agents import AstraAgentSystem
from app.services.commands import CommandService
from app.services.desktop import DesktopActionService
from app.services.llm import NVIDIA_FAST_MODELS, NVIDIA_FREE_CHAT_MODEL_IDS, NVIDIA_PRO_MODELS, LlmService


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


class SlowProFallbackLlm:
    def __init__(self):
        self.models: list[str | None] = []

    def model_for_profile(self, profile: str) -> str:
        return "nvidia:slow-pro" if profile == "pro" else "cerebras:fast-model"

    async def complete(self, _system_prompt: str, _user_prompt: str, model: str | None = None):
        self.models.append(model)
        if model == "nvidia:slow-pro":
            await asyncio.sleep(1)
            return "slow", []
        return "fast fallback", []


class EmptySearch:
    async def search_web(self, query: str, max_results: int = 5):
        return [], []

    async def search_academic(self, query: str, max_results: int = 5):
        return [], []


class EmptySafeAgent:
    async def handle_natural_language(self, text: str, confirmed: bool = False):
        return None


class RecordingSafeAgent:
    def __init__(self, response=None):
        self.calls: list[str] = []
        self.response = response

    async def handle_natural_language(self, text: str, confirmed: bool = False):
        self.calls.append(text)
        return self.response


class FakeAutomationService:
    def __init__(self):
        self.prompts: list[str] = []

    def can_handle_agent_prompt(self, prompt: str) -> bool:
        normalized = prompt.lower()
        return "downloaded" in normalized or "youtube" in normalized or "you tube" in normalized

    async def start_run(self, request):
        self.prompts.append(request.prompt)
        return AutomationRun(id="automation-test", prompt=request.prompt)


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


def test_llm_settings_restrict_cerebras_models_by_profile(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_fast_model="llama3.1-8b",
        cerebras_model="zai-glm-4.7",
        cerebras_pro_model="",
    )
    llm = LlmService(settings)

    response = llm.update_settings(
        LlmSettingsUpdateRequest(
            profiles={
                "fast": LlmProfileConfig(provider="cerebras", model="gpt-oss-120b"),
                "pro": LlmProfileConfig(provider="cerebras", model="llama3.1-8b"),
            }
        )
    )

    assert response.profiles["fast"].model == "llama3.1-8b"
    assert response.profiles["pro"].model == "zai-glm-4.7"


def test_llm_settings_restrict_nvidia_to_free_chat_models(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        nvidia_model="moonshotai/kimi-k2.6",
    )
    llm = LlmService(settings)

    nvidia = next(provider for provider in llm.provider_statuses() if provider.id == "nvidia")

    assert set(nvidia.models) == NVIDIA_FREE_CHAT_MODEL_IDS
    assert "qwen/qwen3-coder-480b-a35b-instruct" not in nvidia.models
    assert "google/gemma-3n-e2b-it" not in nvidia.models
    assert "google/gemma-3n-e4b-it" not in nvidia.models
    assert "z-ai/glm-5.2" not in nvidia.models


def test_llm_settings_restrict_nvidia_models_by_profile(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    llm = LlmService(settings)

    response = llm.update_settings(
        LlmSettingsUpdateRequest(
            profiles={
                "fast": LlmProfileConfig(provider="nvidia", model=NVIDIA_PRO_MODELS[0]),
                "pro": LlmProfileConfig(provider="nvidia", model=NVIDIA_FAST_MODELS[0]),
            }
        )
    )

    assert response.profiles["fast"].model == NVIDIA_FAST_MODELS[0]
    assert response.profiles["pro"].model == NVIDIA_PRO_MODELS[0]


def test_llm_settings_migrates_removed_nvidia_model_to_valid_nvidia_default(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    llm = LlmService(settings)
    llm.llm_settings_path.parent.mkdir(parents=True, exist_ok=True)
    llm.llm_settings_path.write_text(
        '{"profiles":{"pro":{"provider":"nvidia","model":"qwen/qwen3-coder-480b-a35b-instruct"}}}',
        encoding="utf-8",
    )

    profiles = llm.current_profiles()

    assert profiles["pro"].provider == "nvidia"
    assert profiles["pro"].model == NVIDIA_PRO_MODELS[0]


@pytest.mark.asyncio
async def test_cockpit_command_uses_fast_model_by_default(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    await service.handle(CommandRequest(text="summarize my DBMS notes", mode="cockpit", astra_pro=False))

    assert recorder.models == ["fast-model"]


@pytest.mark.asyncio
async def test_cockpit_greeting_returns_without_llm_call(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    response = await service.handle(CommandRequest(text="hiii", mode="cockpit", astra_pro=False))

    assert response.intent == "chat"
    assert response.mode == "cockpit"
    assert response.display_text == "Hello! I'm Astra. How can I help?"
    assert recorder.models == []


@pytest.mark.asyncio
async def test_cockpit_status_check_returns_without_llm_call(tmp_path):
    service, _, recorder = build_command_service(tmp_path)

    response = await service.handle(CommandRequest(text="everything fine ?", mode="cockpit", astra_pro=True))

    assert response.intent == "chat"
    assert response.mode == "cockpit"
    assert response.display_text == "All systems green on my end. How can I help?"
    assert recorder.models == []


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
async def test_agent_mode_delegates_artifact_task_to_automation_runtime(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm()
    agent_system.llm = recorder  # type: ignore[assignment]
    automation = FakeAutomationService()
    service = CommandService(agent_system, DesktopActionService(), object(), EmptySafeAgent(), automation_service=automation)  # type: ignore[arg-type]

    response = await service.handle(CommandRequest(text="play that downloaded song in vlc", mode="agents", astra_pro=False))

    assert response.intent == "agent_plan"
    assert response.suggested_mode == "sources"
    assert response.automation_run is not None
    assert response.automation_run.id == "automation-test"
    assert automation.prompts == ["play that downloaded song in vlc"]
    assert recorder.models == []


@pytest.mark.asyncio
async def test_agent_mode_routes_youtube_media_to_automation_before_safe_open(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"), piper_cache_dir=str(tmp_path / "piper"))
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm()
    agent_system.llm = recorder  # type: ignore[assignment]
    automation = FakeAutomationService()
    safe_agent = RecordingSafeAgent()
    service = CommandService(agent_system, DesktopActionService(), object(), safe_agent, automation_service=automation)  # type: ignore[arg-type]

    response = await service.handle(CommandRequest(text="open youtube and search for codewithharry latest video", mode="agents"))

    assert response.intent == "agent_plan"
    assert response.suggested_mode == "sources"
    assert response.automation_run is not None
    assert automation.prompts == ["open youtube and search for codewithharry latest video"]
    assert safe_agent.calls == []


@pytest.mark.asyncio
async def test_agent_mode_routes_youtube_login_to_safe_form_flow_not_automation(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"), piper_cache_dir=str(tmp_path / "piper"))
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm()
    agent_system.llm = recorder  # type: ignore[assignment]
    form_response = AgentCommandResponse(
        command_id="fill_web_form",
        label="Fill Web Form",
        risk="safe_confirm",
        outcome="confirmation_required",
        message="Review the detected form values before Astra fills the page.",
        confirmation_required=True,
    )
    automation = FakeAutomationService()
    safe_agent = RecordingSafeAgent(form_response)
    service = CommandService(agent_system, DesktopActionService(), object(), safe_agent, automation_service=automation)  # type: ignore[arg-type]

    response = await service.handle(CommandRequest(text="open youtube login page with dummy data", mode="agents"))

    assert response.intent == "agent_command"
    assert response.agent_command is not None
    assert response.agent_command.command_id == "fill_web_form"
    assert automation.prompts == []
    assert safe_agent.calls == ["open youtube login page with dummy data"]


@pytest.mark.asyncio
async def test_agent_mode_routes_youtube_create_account_to_safe_form_flow_not_automation(tmp_path):
    settings = Settings(data_dir=str(tmp_path / "data"), reports_dir=str(tmp_path / "reports"), piper_cache_dir=str(tmp_path / "piper"))
    agent_system = AstraAgentSystem(settings)
    recorder = RecordingLlm()
    agent_system.llm = recorder  # type: ignore[assignment]
    form_response = AgentCommandResponse(
        command_id="fill_web_form",
        label="Fill Web Form",
        risk="safe_confirm",
        outcome="confirmation_required",
        message="Review the detected form values before Astra fills the page.",
        confirmation_required=True,
    )
    automation = FakeAutomationService()
    safe_agent = RecordingSafeAgent(form_response)
    service = CommandService(agent_system, DesktopActionService(), object(), safe_agent, automation_service=automation)  # type: ignore[arg-type]

    response = await service.handle(CommandRequest(text="create youtube account with dummy data", mode="agents"))

    assert response.intent == "agent_command"
    assert response.agent_command is not None
    assert response.agent_command.command_id == "fill_web_form"
    assert automation.prompts == []
    assert safe_agent.calls == ["create youtube account with dummy data"]


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
async def test_cockpit_pro_chat_falls_back_when_selected_model_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.agents.CHAT_LLM_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("app.services.agents.CHAT_LLM_FALLBACK_TIMEOUT_SECONDS", 0.2)
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    agent_system = AstraAgentSystem(settings)
    recorder = SlowProFallbackLlm()
    agent_system.llm = recorder  # type: ignore[assignment]

    response = await agent_system.chat("explain one thing", "cockpit", astra_pro=True)

    assert response.answer == "fast fallback"
    assert response.setup_required == ["LLM_PRO_TIMEOUT_FALLBACK"]
    assert recorder.models == ["nvidia:slow-pro", "cerebras:fast-model"]


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
