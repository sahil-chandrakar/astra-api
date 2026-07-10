import re

from app.models import (
    ActionResult,
    AgentCommandResponse,
    AgentEvent,
    AppMode,
    AutomationRun,
    AutomationRunRequest,
    ChatResponse,
    CommandRequest,
    CommandResponse,
    ResearchDepth,
    ResearchRequest,
    ResearchResponse,
    AutomationSuggestion,
)
from app.services.agents import AstraAgentSystem
from app.services.desktop import DesktopActionService
from app.services.reports import ReportService
from app.services.research import ResearchService
from app.services.safe_agent import SafeAgentService
from app.services.automations import AutomationService


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
        automation_service: AutomationService | None = None,
    ):
        self.agent_system = agent_system
        self.desktop_actions = desktop_actions
        self.reports = reports
        self.safe_agent = safe_agent
        self.research_service = research_service
        self.automation_service = automation_service

    async def handle(self, request: CommandRequest) -> CommandResponse:
        text = request.text.strip()
        requested_mode = self._detect_mode_switch(text)
        if requested_mode:
            return self._mode_switch_response(requested_mode)

        conversational = self._conversation_response(text, request.mode)
        if conversational:
            return conversational

        if request.mode == "agents" and self._should_route_youtube_automation_first(text):
            run = await self.automation_service.start_run(AutomationRunRequest(prompt=text))  # type: ignore[union-attr]
            return self._automation_response(run)

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
            suggestion_handler = getattr(self.automation_service, "automation_suggestion", None) if self.automation_service else None
            if callable(suggestion_handler):
                suggestion = suggestion_handler(text)
                if suggestion:
                    return self._automation_suggestion_response(suggestion, request.mode)
            if self.automation_service and self.automation_service.can_handle_agent_prompt(text):
                run = await self.automation_service.start_run(AutomationRunRequest(prompt=text))
                return self._automation_response(run)
            plan = self.desktop_actions.plan_message(text)
            if plan:
                return self._agent_plan_response(plan, request.mode)
            chat_response = await self.agent_system.chat(text, "agents")
            return self._chat_response(chat_response, request.mode, "chat")

        chat_response = await self.agent_system.chat(text, "cockpit", astra_pro=request.astra_pro)
        return self._chat_response(chat_response, "cockpit", "chat")

    def _should_route_youtube_automation_first(self, text: str) -> bool:
        if not self.automation_service:
            return False
        normalized = self._normalize(text)
        if not re.search(r"\b(youtube|you\s*tube|yt)\b", normalized):
            return False
        if self._is_youtube_account_or_form_intent(normalized):
            return False
        media_intent = re.search(
            r"\b(search|find|play|watch|open\s+.*video|latest|newest|recent|popular|shorts?|live|stream|channel|video|song|title|name|filter|download)\b",
            normalized,
        )
        if not media_intent:
            return False
        can_handle = getattr(self.automation_service, "can_handle_agent_prompt", None)
        return bool(callable(can_handle) and can_handle(text))

    def _is_youtube_account_or_form_intent(self, normalized: str) -> bool:
        form_intent = re.search(r"\b(fill|complete|populate)\b", normalized) and re.search(r"\b(form|login|account|registration|signup)\b", normalized)
        create_account_intent = re.search(
            r"\b(sign\s*up|signup|create\s+(?:an\s+)?(?:\w+\s+){0,4}account|new\s+account|account\s+creation|register|registration)\b",
            normalized,
        )
        login_intent = re.search(r"\b(login|log\s*in|sign\s*in|signin)\b", normalized)
        media_intent = re.search(r"\b(search|find|play|watch|latest|newest|recent|popular|shorts?|live|stream|channel|video|song|title|name|filter|download)\b", normalized)
        explicit_account_page = re.search(r"\b(login|sign\s*in|signin)\s+(?:page|form|screen|account)\b", normalized)
        if form_intent or create_account_intent or explicit_account_page:
            return True
        return bool(login_intent and not media_intent)

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

    def _automation_response(self, run: AutomationRun) -> CommandResponse:
        event = AgentEvent(agent="Automation Agent", status="working", message="Started an intelligent automation workflow.")
        return CommandResponse(
            mode="sources",
            intent="agent_plan",
            suggested_mode="sources",
            spoken_text="I started the automation workflow.",
            display_text="Astra is running this through the shared automation runtime.",
            events=[event],
            automation_run=run,
        )

    def _automation_suggestion_response(self, suggestion: AutomationSuggestion, mode: AppMode) -> CommandResponse:
        recipe = suggestion.recipe
        input_text = ", ".join(suggestion.inputs) if suggestion.inputs else "No required inputs"
        message = (
            f"{suggestion.message}\n"
            f"Engine: OpenRPA\n"
            f"Risk: {recipe.risk.replace('_', ' ')}\n"
            f"Required inputs: {input_text}"
        )
        event = AgentEvent(agent="Automation Agent", status="warning", message=f"Suggested OpenRPA workflow: {recipe.name}.")
        return CommandResponse(
            mode=mode,
            intent="automation_suggestion",
            suggested_mode=mode,
            spoken_text=f"I found a matching OpenRPA workflow: {recipe.name}. Confirm before running it.",
            display_text=message,
            events=[event],
            automation_suggestion=suggestion,
        )

    def _conversation_response(self, text: str, mode: AppMode) -> CommandResponse | None:
        normalized = self._normalize(text)
        compact = re.sub(r"[^a-z0-9]+", "", normalized)
        answer = ""

        if compact in {"hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "hallo", "yo"} or re.fullmatch(r"(hi+|he+y+|hello+|hlo+)", compact):
            answer = "Hello! I'm Astra. How can I help?"
        elif compact in {"whatsup", "sup", "wyd"} or re.fullmatch(r"(what'?s\s+up|what\s+is\s+up|sup)\??", normalized):
            answer = "Not much, just ready to help. What would you like to do?"
        elif re.search(
            r"\b(everything|all|systems?)\s+(fine|good|ok|okay|green|working|normal|nominal)\b", normalized
        ) or re.search(r"\b(are you there|you there|status check|system check)\b", normalized):
            answer = "All systems green on my end. How can I help?"
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
            mode=mode,
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
