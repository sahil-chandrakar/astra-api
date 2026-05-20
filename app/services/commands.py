import re

from app.models import (
    ActionResult,
    AgentCommandResponse,
    AgentEvent,
    AppMode,
    ChatResponse,
    CommandRequest,
    CommandResponse,
    ResearchDepth,
    ResearchRequest,
    ResearchResponse,
)
from app.services.agents import AstraAgentSystem
from app.services.desktop import DesktopActionService
from app.services.reports import ReportService
from app.services.research import ResearchService
from app.services.safe_agent import SafeAgentService


MODE_ALIASES: dict[str, AppMode] = {
    "chat": "cockpit",
    "cockpit": "cockpit",
    "normal": "cockpit",
    "research": "research",
    "deep research": "research",
    "agent": "agents",
    "agents": "agents",
    "source": "sources",
    "sources": "sources",
    "library": "library",
    "timeline": "timeline",
    "settings": "settings",
}


class CommandService:
    def __init__(
        self,
        agent_system: AstraAgentSystem,
        desktop_actions: DesktopActionService,
        reports: ReportService,
        safe_agent: SafeAgentService,
        research_service: ResearchService | None = None,
    ):
        self.agent_system = agent_system
        self.desktop_actions = desktop_actions
        self.reports = reports
        self.safe_agent = safe_agent
        self.research_service = research_service

    async def handle(self, request: CommandRequest) -> CommandResponse:
        text = request.text.strip()
        requested_mode = self._detect_mode_switch(text)
        if requested_mode:
            return self._mode_switch_response(requested_mode)

        if request.mode == "agents" or self._looks_like_mock_test_intent(text):
            agent_response = await self.safe_agent.handle_natural_language(text, confirmed=request.confirmed)
            if agent_response and (request.mode == "agents" or agent_response.command_id in {"generate_mock_test", "open_latest_mock_test", "list_mock_tests"}):
                return self._agent_command_response(agent_response, request.mode)

        desktop_target = self.desktop_actions.detect(text)
        if desktop_target:
            return self._desktop_response(await self.desktop_actions.execute(text), request.mode)

        if self._should_research(text, request.mode):
            return await self._research_response(request)

        if request.mode == "agents":
            conversational = self._agent_conversation_response(text)
            if conversational:
                return conversational
            plan = self.desktop_actions.plan_message(text)
            if plan:
                return self._agent_plan_response(plan, request.mode)
            chat_response = await self.agent_system.chat(text, "agents")
            return self._chat_response(chat_response, request.mode, "chat")

        chat_response = await self.agent_system.chat(text, "cockpit", astra_pro=request.astra_pro)
        return self._chat_response(chat_response, "cockpit", "chat")

    def _mode_switch_response(self, mode: AppMode) -> CommandResponse:
        label = "cockpit chat" if mode == "cockpit" else mode.replace("_", " ")
        event = AgentEvent(agent="Command Router", status="complete", message=f"Mode switched to {label}.")
        return CommandResponse(
            mode=mode,
            intent="mode_switch",
            suggested_mode=mode,
            spoken_text=f"Switched to {label} mode.",
            display_text=f"Astra is now in {label} mode.",
            events=[event],
        )

    def _desktop_response(self, result: ActionResult, mode: AppMode) -> CommandResponse:
        status = "complete" if result.ok else "warning"
        event = AgentEvent(agent="Desktop Agent", status=status, message=result.message)
        return CommandResponse(
            mode=mode,
            intent="desktop_action",
            spoken_text=result.message,
            display_text=result.message,
            action_result=result,
            events=[event],
        )

    def _agent_plan_response(self, result: ActionResult, mode: AppMode) -> CommandResponse:
        event = AgentEvent(agent="Supervisor", status="warning", message="Plan prepared; execution needs an allowlisted action.")
        return CommandResponse(
            mode=mode,
            intent="agent_plan",
            spoken_text="I prepared the plan, but I did not execute anything outside the safe allowlist.",
            display_text=result.message,
            action_result=result,
            events=[event],
        )

    def _agent_command_response(self, response: AgentCommandResponse, mode: AppMode) -> CommandResponse:
        spoken = response.message
        return CommandResponse(
            mode=mode,
            intent="agent_command",
            spoken_text=spoken,
            display_text=response.message,
            events=response.events,
            agent_command=response,
        )

    def _agent_conversation_response(self, text: str) -> CommandResponse | None:
        normalized = self._normalize(text)
        compact = re.sub(r"[^a-z0-9]+", "", normalized)
        answer = ""

        if compact in {"hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "hallo", "yo"} or re.fullmatch(r"(hi+|he+y+|hello+|hlo+)", compact):
            answer = "Hello! I'm Astra. How can I help?"
        elif re.search(r"\b(what is your name|what's your name|who are you|your name)\b", normalized):
            answer = "My name is Astra. I'm your voice-first AI assistant."
        elif re.search(r"\b(how are you|how r you|how are u)\b", normalized):
            answer = "I'm ready and online. What would you like to do?"
        elif compact in {"thanks", "thankyou", "thanku", "ty"} or re.search(r"\b(thanks|thank you)\b", normalized):
            answer = "You're welcome."

        if not answer:
            return None

        event = AgentEvent(agent="Astra", status="complete", message="Conversational response ready.")
        return CommandResponse(
            mode="agents",
            intent="chat",
            spoken_text=answer,
            display_text=answer,
            events=[event],
        )

    def _chat_response(self, response: ChatResponse, mode: AppMode, intent: str) -> CommandResponse:
        spoken = self._spoken_chat(response.answer)
        return CommandResponse(
            mode=mode,
            intent=intent,  # type: ignore[arg-type]
            spoken_text=spoken,
            display_text=response.answer,
            events=response.events,
            setup_required=response.setup_required,
        )

    async def _research_response(self, request: CommandRequest) -> CommandResponse:
        depth = self._research_depth(request)
        topic = self._clean_research_topic(request.text)
        research_request = ResearchRequest(topic=topic, depth=depth, source_mode="mixed", require_citations=True)
        if self.research_service:
            job = await self.research_service.run_to_completion(research_request, save_report=True)
            response: ResearchResponse = self.research_service.to_research_response(job)
            report = job.report
        else:
            response = await self.agent_system.research(research_request)
            report = self.reports.save_research_report(research_request, response)
        percent = round(response.confidence * 100)
        source_count = len(response.citations)
        spoken = f"Research complete. I generated the report with {source_count} sources and {percent}% confidence."
        display = f"{spoken} Report saved: {report.title if report else topic}"
        return CommandResponse(
            mode="research",
            intent="research",
            spoken_text=spoken,
            display_text=display,
            events=response.events,
            citations=response.citations,
            confidence=response.confidence,
            critic_notes=response.critic_notes,
            setup_required=response.setup_required,
            report=report,
        )

    def _detect_mode_switch(self, text: str) -> AppMode | None:
        normalized = self._normalize(text)
        if not re.search(r"\b(switch|change|go|move|set)\b", normalized):
            return None
        for alias, mode in MODE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return mode
        return None

    def _should_research(self, text: str, mode: AppMode) -> bool:
        normalized = self._normalize(text)
        return bool(
            re.search(
                r"\b(research|deep research|web search|search web|latest|sources?|citations?|papers?|study|studies|literature review|academic)\b",
                normalized,
            )
        )

    def _looks_like_mock_test_intent(self, text: str) -> bool:
        normalized = self._normalize(text)
        if re.search(r"\b(mock|mocktest|practice test|mock exam)\b", normalized):
            return True
        if re.search(r"\b(test|quiz|assess|challenge)\s+(me|my knowledge)?\b", normalized):
            return not re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized)
        if re.search(r"\b(ask|give)\s+(me\s+)?(some\s+|a\s+)?(questions?|mcqs?|quiz)\b", normalized):
            return True
        if re.search(r"\btest\b", normalized) and re.search(r"\b(create|generate|make|build|prepare|new|another)\b", normalized):
            return not re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized)
        return False

    def _research_depth(self, request: CommandRequest) -> ResearchDepth:
        normalized = self._normalize(request.text)
        if request.depth:
            return request.depth
        if "deep" in normalized or "literature review" in normalized or request.mode == "research":
            return "deep"
        return "quick"

    def _clean_research_topic(self, text: str) -> str:
        topic = re.sub(
            r"^\s*(please\s+)?(do\s+)?(a\s+)?(deep\s+)?(research|search|web search|find|look up|literature review)\s+(on|about|for)?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        return topic.strip(" .") or text.strip()

    def _spoken_chat(self, answer: str) -> str:
        clean = re.sub(r"\s+", " ", answer).strip()
        if len(clean) <= 420:
            return clean
        match = re.search(r"^(.{160,420}?[.!?])\s", clean)
        return (match.group(1) if match else clean[:360].rstrip()) + " I wrote the rest in the transcript."

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()
