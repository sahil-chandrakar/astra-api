from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AgentStatus = Literal["idle", "working", "complete", "warning", "error"]
ResearchDepth = Literal["quick", "deep", "academic"]
ResearchJobStatus = Literal["queued", "planning", "searching", "reading", "extracting", "verifying", "writing", "complete", "error"]
ResearchSourcePolicy = Literal["latest_web_first", "broad_web_academic", "academic_first"]
AppMode = Literal["cockpit", "research", "agents", "sources", "library", "timeline", "settings"]
AgentCommandRisk = Literal["safe_auto", "safe_confirm", "blocked"]
AgentCommandTestStatus = Literal["untested", "passed", "failed"]
AgentCommandOutcome = Literal["success", "failure", "blocked", "confirmation_required", "planned"]
AutomationRunStatus = Literal["queued", "planning", "running", "waiting_for_login", "waiting_for_user", "confirmation_required", "complete", "error", "cancelled"]
AgentMemoryCategory = Literal["course", "project", "goal", "preference", "general"]
StudyArtifactType = Literal["notes", "flashcards", "quiz", "revision_plan", "viva_questions"]
MockTestDifficulty = Literal["easy", "medium", "hard", "mixed"]
MockTestMode = Literal["mcq"]
MockTestSourceRequirement = Literal["none", "pyq_required", "source_backed"]
MockTestSourceMode = Literal["web", "uploaded_docs", "mixed"]
MockTestGenerationMode = Literal["topic_practice", "profile_based", "syllabus_based", "source_backed_pyq", "pyq_style", "llm_planned"]
MockAttemptStatus = Literal["active", "submitted"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = "general"
    context: list[str] = Field(default_factory=list)
    astra_pro: bool = False


class Source(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str = "unknown"
    domain: str = ""
    published_at: str | None = None
    fetched_chars: int = 0
    quality_score: float = 0.0
    extraction_status: str = ""


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
    depth: ResearchDepth = "deep"
    source_mode: Literal["web", "academic", "mixed"] = "mixed"
    source_policy: ResearchSourcePolicy = "latest_web_first"
    max_candidates: int = Field(default=60, ge=10, le=100)
    max_sources: int = Field(default=20, ge=3, le=30)
    recency_days: int | None = Field(default=365, ge=1, le=3650)
    require_citations: bool = True


class ResearchReport(BaseModel):
    id: str
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    markdown: str = ""
    download_url: str = ""


class ResearchResponse(BaseModel):
    summary: str
    detailed_answer: str
    citations: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    critic_notes: list[str] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)
    job_id: str | None = None
    status: ResearchJobStatus | None = None
    report: ResearchReport | None = None


class ResearchJobResponse(BaseModel):
    id: str
    status: ResearchJobStatus = "queued"
    request: ResearchRequest
    summary: str = ""
    detailed_answer: str = ""
    citations: list[Source] = Field(default_factory=list)
    confidence: float = 0.0
    critic_notes: list[str] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)
    report: ResearchReport | None = None
    error: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


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
    astra_pro: bool = False


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


class AutomationEvent(BaseModel):
    id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class AutomationRun(BaseModel):
    id: str
    prompt: str
    status: AutomationRunStatus = "queued"
    current_url: str = ""
    events: list[AutomationEvent] = Field(default_factory=list)
    result: str = ""
    error: str = ""
    recipe_id: str | None = None
    create_recipe: bool = False
    confirmation: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AutomationRecipe(BaseModel):
    id: str
    name: str
    prompt: str
    steps: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AutomationRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    recipe_id: str | None = None
    create_recipe: bool = False


class AutomationRecipeCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    steps: list[dict[str, Any]] = Field(default_factory=list)


class AutomationContinueRequest(BaseModel):
    note: str = ""


class AutomationCancelRequest(BaseModel):
    note: str = ""


class AutomationConfirmRequest(BaseModel):
    approved: bool
    confirmed_rights: bool = False
    attestation: str = ""


class AutomationOpenPathRequest(BaseModel):
    path: str = Field(..., min_length=1)


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


class MockQuestion(BaseModel):
    id: str
    prompt: str = Field(..., min_length=1)
    options: list[str] = Field(..., min_length=4, max_length=4)
    correct_option_index: int = Field(..., ge=0, le=3)
    explanation: str = Field(..., min_length=1)
    difficulty: MockTestDifficulty = "medium"
    tags: list[str] = Field(default_factory=list)
    source_refs: list[Source] = Field(default_factory=list)


class MockQuestionView(BaseModel):
    id: str
    prompt: str
    options: list[str]
    difficulty: MockTestDifficulty = "medium"
    tags: list[str] = Field(default_factory=list)
    source_refs: list[Source] = Field(default_factory=list)


class MockQuestionReview(MockQuestionView):
    correct_option_index: int
    selected_option_index: int | None = None
    is_correct: bool = False
    explanation: str


class MockTest(BaseModel):
    id: str
    topic: str = Field(..., min_length=1)
    exam: str = ""
    subject: str = ""
    mode: MockTestMode = "mcq"
    difficulty: MockTestDifficulty = "mixed"
    question_count: int = Field(default=10, ge=1, le=50)
    duration_minutes: int = Field(default=20, ge=1, le=180)
    questions: list[MockQuestion]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "llm"
    generation_mode: MockTestGenerationMode = "topic_practice"
    blueprint_source: str = ""
    syllabus_units: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    quality_warnings: list[str] = Field(default_factory=list)
    source_requirement: MockTestSourceRequirement = "none"
    source_mode: MockTestSourceMode = "uploaded_docs"
    source_query: str = ""
    constraints: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)


class MockTestView(BaseModel):
    id: str
    topic: str
    exam: str = ""
    subject: str = ""
    mode: MockTestMode = "mcq"
    difficulty: MockTestDifficulty = "mixed"
    question_count: int = 10
    duration_minutes: int = 20
    questions: list[MockQuestionView]
    created_at: datetime
    source: str = "llm"
    generation_mode: MockTestGenerationMode = "topic_practice"
    blueprint_source: str = ""
    syllabus_units: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    quality_warnings: list[str] = Field(default_factory=list)
    source_requirement: MockTestSourceRequirement = "none"
    source_mode: MockTestSourceMode = "uploaded_docs"
    source_query: str = ""
    constraints: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    setup_required: list[str] = Field(default_factory=list)


class MockTestGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    exam: str = ""
    subject: str = ""
    question_count: int = Field(default=10, ge=1, le=50)
    difficulty: MockTestDifficulty = "mixed"
    mode: MockTestMode = "mcq"
    duration_minutes: int = Field(default=20, ge=1, le=180)
    source_requirement: MockTestSourceRequirement = "none"
    source_mode: MockTestSourceMode = "uploaded_docs"
    source_query: str = ""
    constraints: list[str] = Field(default_factory=list)
    source_text: str = ""
    source_refs: list[Source] = Field(default_factory=list)


class MockTestGenerateResponse(BaseModel):
    test: MockTestView
    setup_required: list[str] = Field(default_factory=list)


class MockAttempt(BaseModel):
    id: str
    test_id: str
    status: MockAttemptStatus = "active"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    elapsed_seconds: int = 0
    answers: dict[str, int] = Field(default_factory=dict)


class MockTestStartResponse(BaseModel):
    test: MockTestView
    attempt: MockAttempt


class MockTestSubmitRequest(BaseModel):
    answers: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: int = Field(default=0, ge=0)


class MockTestSubmitResponse(BaseModel):
    test: MockTestView
    attempt: MockAttempt
    score: int
    total: int
    percentage: float
    correct_count: int
    incorrect_count: int
    elapsed_seconds: int
    review: list[MockQuestionReview]


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
