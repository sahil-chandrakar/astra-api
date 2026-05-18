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
    ):
        self.agent_system = agent_system
        self.desktop_actions = desktop_actions
        self.reports = reports
        self.safe_agent = safe_agent

    async def handle(self, request: CommandRequest) -> CommandResponse:
        text = request.text.strip()
        requested_mode = self._detect_mode_switch(text)
        if requested_mode:
            return self._mode_switch_response(requested_mode)

        if request.mode == "agents":
            agent_response = await self.safe_agent.handle_natural_language(text, confirmed=request.confirmed)
            if agent_response:
                return self._agent_command_response(agent_response, request.mode)

        desktop_target = self.desktop_actions.detect(text)
        if desktop_target:
            return self._desktop_response(await self.desktop_actions.execute(text), request.mode)

        if self._should_research(text, request.mode):
            return await self._research_response(request)

        if request.mode == "agents":
            plan = self.desktop_actions.plan_message(text)
            if plan:
                return self._agent_plan_response(plan, request.mode)
            chat_response = await self.agent_system.chat(text, "agents")
            return self._chat_response(chat_response, request.mode, "agent_plan")

        chat_response = await self.agent_system.chat(text, "cockpit")
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
        response: ResearchResponse = await self.agent_system.research(research_request)
        report = self.reports.save_research_report(research_request, response)
        percent = round(response.confidence * 100)
        source_count = len(response.citations)
        spoken = f"Research complete. I generated the report with {source_count} sources and {percent}% confidence."
        display = f"{spoken} Report saved: {report.title}"
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
        if mode == "research":
            return True
        return bool(
            re.search(
                r"\b(research|deep research|web search|search web|latest|sources?|citations?|papers?|study|studies|literature review|academic)\b",
                normalized,
            )
        )

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
