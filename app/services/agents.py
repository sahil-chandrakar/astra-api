from app.config import Settings
from app.models import AgentDescriptor, AgentEvent, ChatResponse, ResearchRequest, ResearchResponse, Source
from app.services.llm import LlmService
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

    def descriptors(self) -> list[AgentDescriptor]:
        return AGENTS

    async def chat(self, message: str, mode: str = "general") -> ChatResponse:
        events = [
            AgentEvent(agent="Command Router", status="complete", message=f"Routing {mode} request."),
            AgentEvent(agent="Writer Agent", status="working", message="Preparing Astra response."),
        ]
        system_prompt = (
            "You are Astra, a fast voice-first college AI agent. "
            "In cockpit chat, answer in 2-4 practical sentences unless the user asks for depth."
        )
        if mode == "agents":
            system_prompt = (
                "You are Astra in agent mode. Think like a supervisor: identify intent, give a short plan, "
                "name what can be executed now, and avoid pretending to control the desktop unless a tool result says it happened."
            )
        answer, setup = await self.llm.complete(
            system_prompt,
            message,
        )
        events.append(AgentEvent(agent="Writer Agent", status="complete", message="Response ready."))
        return ChatResponse(answer=answer, events=events, setup_required=setup)

    async def research(self, request: ResearchRequest) -> ResearchResponse:
        events: list[AgentEvent] = [
            AgentEvent(agent="Query", status="complete", message=f"Research request received: {request.topic}"),
            AgentEvent(agent="Supervisor", status="complete", message=f"Research plan created for: {request.topic}"),
            AgentEvent(agent="Search Agents", status="working", message="Searching selected providers."),
        ]
        setup_required: list[str] = []
        sources: list[Source] = []

        if request.source_mode in {"web", "mixed"}:
            web_sources, web_setup = await self.search.search_web(request.topic, max_results=5)
            sources.extend(web_sources)
            setup_required.extend(web_setup)

        if request.source_mode in {"academic", "mixed"}:
            academic_sources, academic_setup = await self.search.search_academic(request.topic, max_results=5)
            sources.extend(academic_sources)
            setup_required.extend(academic_setup)

        sources = self._dedupe_sources(sources)[:8]
        events.append(
            AgentEvent(
                agent="Search Agents",
                status="complete" if sources else "warning",
                message=f"Found {len(sources)} usable sources.",
                sources=sources[:3],
            )
        )
        events.append(AgentEvent(agent="Reader Agents", status="working", message="Extracting useful evidence."))

        evidence = "\n".join(
            f"- {source.title} ({source.provider}): {source.snippet[:500]} URL: {source.url}"
            for source in sources
        )
        if not evidence:
            evidence = "No live sources were available. Explain the setup needed and provide a safe research outline."

        events.append(AgentEvent(agent="Reader Agents", status="complete", message="Evidence extracted.", sources=sources[:3]))
        events.append(AgentEvent(agent="Citation Agent", status="complete", message="Citations organized.", sources=sources))
        events.append(AgentEvent(agent="Final Boss", status="working", message="Checking source quality and gaps."))

        system_prompt = (
            "You are Astra's multi-agent research writer. Use the provided evidence first. "
            "Be honest about weak or missing sources. Return a useful college-project style answer."
        )
        user_prompt = (
            f"Topic: {request.topic}\nDepth: {request.depth}\n"
            f"Evidence:\n{evidence}\n\n"
            "Write a concise summary, a detailed answer, and mention source confidence."
        )
        llm_answer, llm_setup = await self.llm.complete(system_prompt, user_prompt)
        setup_required.extend(llm_setup)

        critic_notes = self._critic_notes(sources, setup_required)
        confidence = 0.78 if sources and not llm_setup else 0.42 if sources else 0.18

        events.append(AgentEvent(agent="Final Boss", status="complete", message="Quality check complete."))
        events.append(AgentEvent(agent="Writer Agent", status="complete", message="Final research answer ready."))
        events.append(AgentEvent(agent="Complete", status="complete", message="Research workflow complete."))

        return ResearchResponse(
            summary=self._summary_from_answer(llm_answer),
            detailed_answer=llm_answer,
            citations=sources,
            confidence=confidence,
            critic_notes=critic_notes,
            events=events,
            setup_required=sorted(set(setup_required)),
        )

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
