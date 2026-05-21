import re

from app.config import Settings
from app.models import AgentDescriptor, AgentEvent, ChatResponse, ResearchRequest, ResearchResponse, Source
from app.services.llm import LlmService
from app.services.reports import ReportService
from app.services.research import ResearchService
from app.services.search import SearchService


AGENTS = [
    AgentDescriptor(name="Supervisor", role="Breaks the task into agent steps."),
    AgentDescriptor(name="Search Agents", role="Gather web, academic, and fallback sources."),
    AgentDescriptor(name="Reader Agents", role="Extract useful facts from discovered sources."),
    AgentDescriptor(name="Citation Agent", role="Keeps source links and evidence organized."),
    AgentDescriptor(name="Final Boss", role="Checks quality, contradictions, and missing proof."),
    AgentDescriptor(name="Writer Agent", role="Produces the final answer for Astra."),
    AgentDescriptor(name="Voice Agent", role="Coordinates listening, speech, and voice state."),
]


class AstraAgentSystem:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LlmService(settings)
        self.search = SearchService(settings)
        self.research_service = ResearchService(settings, self.llm, self.search, ReportService(settings))

    def descriptors(self) -> list[AgentDescriptor]:
        return AGENTS

    async def chat(self, message: str, mode: str = "general", astra_pro: bool = False) -> ChatResponse:
        events = [
            AgentEvent(agent="Command Router", status="complete", message=f"Routing {mode} request."),
            AgentEvent(agent="Writer Agent", status="working", message="Preparing Astra response."),
        ]
        profile = "pro" if astra_pro or mode == "agents" else "fast"
        model_selector = getattr(self.llm, "model_for_profile", None)
        model = model_selector(profile) if callable(model_selector) else (
            self.settings.resolved_cerebras_pro_model if profile == "pro" else self.settings.resolved_cerebras_fast_model
        )
        system_prompt = (
            "You are Astra, a fast voice-first college AI agent. "
            "In cockpit chat, answer in 2-4 practical sentences unless the user asks for depth."
        )
        if mode == "agents":
            system_prompt = (
                "You are Astra in agent mode. Answer the user naturally in 1-3 concise sentences. "
                "Do not show internal labels such as Intent, Plan, Execution, Executable Now, or tool reasoning unless the user explicitly asks for a plan. "
                "Do not claim that desktop or file actions happened unless a tool result says they happened. "
                "For unsupported actions, briefly explain that Astra can only execute allowlisted safe commands."
            )
        answer, setup = await self.llm.complete(
            system_prompt,
            message,
            model=model,
        )
        if mode == "agents":
            answer = self._clean_agent_chat_answer(answer)
        events.append(AgentEvent(agent="Writer Agent", status="complete", message="Response ready."))
        return ChatResponse(answer=answer, events=events, setup_required=setup)

    async def research(self, request: ResearchRequest) -> ResearchResponse:
        self.research_service.llm = self.llm
        self.research_service.search = self.search
        job = await self.research_service.run_to_completion(request, save_report=False)
        return self.research_service.to_research_response(job)

    def _critic_notes(self, sources: list[Source], setup_required: list[str]) -> list[str]:
        notes: list[str] = []
        if not sources:
            notes.append("No live sources were retrieved; add Tavily or check network access for stronger research.")
        if "CEREBRAS_API_KEY" in setup_required:
            notes.append("Cerebras key is missing, so Astra is using setup-mode text instead of LLM synthesis.")
        if len(sources) < 3:
            notes.append("Use deep research mode when you need stronger citation coverage.")
        if not notes:
            notes.append("Sources are available and the answer was synthesized with the configured model.")
        return notes

    def _summary_from_answer(self, answer: str) -> str:
        first = answer.split("\n", 1)[0].strip()
        return first[:260] if first else "Research workflow completed."

    def _dedupe_sources(self, sources: list[Source]) -> list[Source]:
        seen: set[str] = set()
        unique: list[Source] = []
        for source in sources:
            key = source.url.lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(source)
        return unique

    def _clean_agent_chat_answer(self, answer: str) -> str:
        if not answer:
            return answer

        lines = [line.strip() for line in answer.splitlines()]
        cleaned: list[str] = []
        for line in lines:
            if not line:
                continue
            if re.match(r"^\**\s*(intent|plan|executable now)\s*:\**", line, flags=re.IGNORECASE):
                continue
            execution = re.match(r"^\**\s*(execution|response)\s*:\**\s*(.+)$", line, flags=re.IGNORECASE)
            if execution:
                cleaned.append(execution.group(2).strip())
                continue
            cleaned.append(line)

        result = "\n".join(cleaned).strip()
        return result or answer.strip()
