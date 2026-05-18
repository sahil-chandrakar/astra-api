from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentStatus = Literal["idle", "working", "complete", "warning", "error"]
ResearchDepth = Literal["quick", "deep", "academic"]
AppMode = Literal["cockpit", "research", "agents", "sources", "library", "timeline", "settings"]
AgentCommandRisk = Literal["safe_auto", "safe_confirm", "blocked"]
AgentCommandTestStatus = Literal["untested", "passed", "failed"]
AgentCommandOutcome = Literal["success", "failure", "blocked", "confirmation_required", "planned"]
AgentMemoryCategory = Literal["course", "project", "goal", "preference", "general"]
StudyArtifactType = Literal["notes", "flashcards", "quiz", "revision_plan", "viva_questions"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = "general"
    context: list[str] = Field(default_factory=list)


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str = "unknown"


class AgentEvent(BaseModel):
    agent: str
    status: AgentStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: list[Source] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    events: list[AgentEvent] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=2)
    depth: ResearchDepth = "quick"
    source_mode: Literal["web", "academic", "mixed"] = "mixed"
    require_citations: bool = True


class ResearchResponse(BaseModel):
    summary: str
    detailed_answer: str
    citations: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    critic_notes: list[str] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    markdown: str = ""
    download_url: str = ""


class ActionResult(BaseModel):
    ok: bool
    action: str
    target: str
    message: str


class CommandRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: AppMode = "cockpit"
    input_source: Literal["typed", "voice", "quick_action"] = "typed"
    depth: ResearchDepth | None = None
    confirmed: bool = False


class CommandResponse(BaseModel):
    mode: AppMode
    intent: Literal["chat", "research", "desktop_action", "mode_switch", "agent_plan", "agent_command"]
    spoken_text: str
    display_text: str
    events: list[AgentEvent] = Field(default_factory=list)
    citations: list[Source] = Field(default_factory=list)
    confidence: float | None = None
    critic_notes: list[str] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)
    report: ResearchReport | None = None
    action_result: ActionResult | None = None
    suggested_mode: AppMode | None = None
    agent_command: "AgentCommandResponse | None" = None


class AgentDescriptor(BaseModel):
    name: str
    role: str
    status: AgentStatus = "idle"


class AgentAbility(BaseModel):
    id: str
    label: str
    description: str
    category: str
    risk: AgentCommandRisk
    params_schema: dict[str, Any] = Field(default_factory=dict)
    test_status: AgentCommandTestStatus = "untested"
    last_tested_at: datetime | None = None
    test_message: str = ""


class AgentCommandRequest(BaseModel):
    command_id: str
    input_text: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False
    resolution: dict[str, Any] = Field(default_factory=dict)


class AgentAuditEntry(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    command_id: str
    label: str = ""
    input_text: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    outcome: AgentCommandOutcome
    safety_decision: str
    message: str
    resolution: dict[str, Any] = Field(default_factory=dict)


class AgentCommandResponse(BaseModel):
    command_id: str
    label: str
    risk: AgentCommandRisk
    outcome: AgentCommandOutcome
    message: str
    confirmation_required: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    events: list[AgentEvent] = Field(default_factory=list)
    audit: AgentAuditEntry | None = None
    resolution: dict[str, Any] = Field(default_factory=dict)


class AgentCommandTestRequest(BaseModel):
    command_id: str | None = None


class AgentCommandTestResponse(BaseModel):
    abilities: list[AgentAbility]
    events: list[AgentEvent] = Field(default_factory=list)


class AgentMemoryItem(BaseModel):
    id: str
    category: AgentMemoryCategory = "general"
    text: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AgentMemoryCreateRequest(BaseModel):
    category: AgentMemoryCategory = "general"
    text: str = Field(..., min_length=1)


class AgentMemoryUpdateRequest(BaseModel):
    category: AgentMemoryCategory | None = None
    text: str | None = Field(default=None, min_length=1)


class DocumentRecord(BaseModel):
    id: str
    title: str
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    page_count: int = 0
    text_preview: str = ""


class DocumentQuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class DocumentQuestionResponse(BaseModel):
    document: DocumentRecord
    answer: str
    page_refs: list[int] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)


class StudyArtifact(BaseModel):
    id: str
    artifact_type: StudyArtifactType
    title: str
    source: str = ""
    markdown: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StudyGenerateRequest(BaseModel):
    artifact_type: StudyArtifactType
    topic: str = Field(..., min_length=1)
    source_text: str = ""
    report_id: str | None = None
    document_id: str | None = None


class VoiceTranscriptionResponse(BaseModel):
    transcript: str = ""
    message: str
    setup_required: list[str] = Field(default_factory=list)


class VoiceSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1400)
    voice: str | None = None


class VoiceWarmupRequest(BaseModel):
    voice: str | None = None


class VoiceStatusResponse(BaseModel):
    enabled: bool
    provider: str = "piper"
    voice: str
    cached: bool = False
    loaded: bool = False
    setup_required: list[str] = Field(default_factory=list)
    message: str = ""
