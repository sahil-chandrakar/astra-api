import asyncio

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import get_settings
from app.models import (
    AgentCommandRequest,
    AgentCommandTestRequest,
    AgentMemoryCreateRequest,
    AgentMemoryUpdateRequest,
    AutomationCancelRequest,
    AutomationConfirmRequest,
    AutomationContinueRequest,
    AutomationOpenPathRequest,
    AutomationRecipeCreateRequest,
    AutomationRunRequest,
    ChatRequest,
    CommandRequest,
    DocumentQuestionRequest,
    LlmSettingsUpdateRequest,
    MockTestGenerateRequest,
    MockTestSubmitRequest,
    ResearchRequest,
    StudyGenerateRequest,
    VoiceSpeakRequest,
    VoiceWarmupRequest,
    VoiceTranscriptionResponse,
)
from app.services.agents import AstraAgentSystem
from app.services.automations import AutomationService
from app.services.commands import CommandService
from app.services.desktop import DesktopActionService
from app.services.documents import DocumentService
from app.services.memory import MemoryService
from app.services.mock_tests import MockTestService
from app.services.reports import ReportService
from app.services.research import ResearchService
from app.services.safe_agent import SafeAgentService
from app.services.study import StudyService
from app.services.voice import VoiceService

settings = get_settings()
agent_system = AstraAgentSystem(settings)
voice_service = VoiceService(settings)
report_service = ReportService(settings)
research_service = ResearchService(settings, agent_system.llm, agent_system.search, report_service)
agent_system.research_service = research_service
desktop_service = DesktopActionService()
memory_service = MemoryService(settings)
document_service = DocumentService(settings)
study_service = StudyService(settings, report_service, document_service, agent_system.llm)
mock_test_service = MockTestService(settings, agent_system.llm, document_service, agent_system.search)
safe_agent_service = SafeAgentService(settings, report_service, memory_service, document_service, study_service, mock_test_service, desktop_service, voice_service)
automation_service = AutomationService(settings, agent_system.llm)
command_service = CommandService(agent_system, desktop_service, report_service, safe_agent_service, research_service)

app = FastAPI(title="Astra AI Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_voice() -> None:
    if voice_service.status().enabled:
        async def run_warmup() -> None:
            try:
                await voice_service.warm_up()
            except Exception:
                pass

        asyncio.create_task(run_warmup())


@app.get("/health")
async def health() -> dict[str, object]:
    llm_profiles = agent_system.llm.current_profiles()
    return {
        "status": "ok",
        "app": "Astra AI Agent API",
        "model": agent_system.llm.model_for_profile("pro"),
        "models": {
            "fast": agent_system.llm.model_for_profile("fast"),
            "pro": agent_system.llm.model_for_profile("pro"),
        },
        "llm_profiles": {name: profile.model_dump(mode="json") for name, profile in llm_profiles.items()},
        "providers": {
            "cerebras": settings.has_cerebras,
            "nvidia": settings.has_nvidia,
            "tavily": settings.has_tavily,
            "openalex": True,
            "semantic_scholar": True,
            "duckduckgo_fallback": True,
            "piper_local": voice_service.status().enabled,
        },
    }


@app.get("/api/llm/settings")
async def llm_settings():
    return agent_system.llm.settings_response()


@app.put("/api/llm/settings")
async def update_llm_settings(request: LlmSettingsUpdateRequest):
    return agent_system.llm.update_settings(request)


@app.get("/api/agents")
async def agents():
    return agent_system.descriptors()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return await agent_system.chat(request.message, request.mode, astra_pro=request.astra_pro)


@app.post("/api/research")
async def research(request: ResearchRequest):
    job = await research_service.run_to_completion(request, save_report=False)
    return research_service.to_research_response(job)


@app.post("/api/research/jobs")
async def start_research_job(request: ResearchRequest):
    return research_service.start_job(request)


@app.get("/api/research/jobs/{job_id}")
async def get_research_job(job_id: str):
    job = research_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Research job not found.")
    return job


@app.get("/api/research/jobs/{job_id}/events")
async def research_job_events(job_id: str):
    if not research_service.get_job(job_id):
        raise HTTPException(status_code=404, detail="Research job not found.")
    return StreamingResponse(research_service.stream_events(job_id), media_type="text/event-stream")


@app.post("/api/command")
async def command(request: CommandRequest):
    return await command_service.handle(request)


@app.get("/api/agent/abilities")
async def agent_abilities():
    return safe_agent_service.abilities()


@app.post("/api/agent/execute")
async def execute_agent_command(request: AgentCommandRequest):
    return await safe_agent_service.execute(request)


@app.post("/api/agent/commands/test")
async def test_agent_commands(request: AgentCommandTestRequest | None = None):
    return await safe_agent_service.test_commands(request.command_id if request else None)


@app.get("/api/agent/audit")
async def agent_audit():
    return safe_agent_service.audit_entries()


@app.post("/api/automations/runs")
async def start_automation_run(request: AutomationRunRequest):
    return await automation_service.start_run(request)


@app.get("/api/automations/runs/{run_id}")
async def get_automation_run(run_id: str):
    run = automation_service.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Automation run not found.")
    return run


@app.get("/api/automations/runs/{run_id}/events")
async def automation_run_events(run_id: str):
    if not automation_service.get_run(run_id):
        raise HTTPException(status_code=404, detail="Automation run not found.")
    return StreamingResponse(automation_service.stream_events(run_id), media_type="text/event-stream")


@app.post("/api/automations/runs/{run_id}/continue")
async def continue_automation_run(run_id: str, request: AutomationContinueRequest):
    run = await automation_service.continue_run(run_id, request)
    if not run:
        raise HTTPException(status_code=404, detail="Automation run not found.")
    return run


@app.post("/api/automations/runs/{run_id}/confirm")
async def confirm_automation_run(run_id: str, request: AutomationConfirmRequest):
    run = await automation_service.confirm_run(run_id, request)
    if not run:
        raise HTTPException(status_code=404, detail="Automation run not found.")
    return run


@app.post("/api/automations/runs/{run_id}/cancel")
async def cancel_automation_run(run_id: str, request: AutomationCancelRequest):
    run = await automation_service.cancel_run(run_id, request)
    if not run:
        raise HTTPException(status_code=404, detail="Automation run not found.")
    return run


@app.post("/api/automations/open-download-folder")
async def open_automation_download_folder(request: AutomationOpenPathRequest):
    if not automation_service.open_download_path(request.path):
        raise HTTPException(status_code=404, detail="Download folder not found.")
    return {"ok": True}


@app.get("/api/automations/recipes")
async def automation_recipes():
    return automation_service.list_recipes()


@app.post("/api/automations/recipes")
async def create_automation_recipe(request: AutomationRecipeCreateRequest):
    return automation_service.create_recipe(request)


@app.delete("/api/automations/recipes/{recipe_id}")
async def delete_automation_recipe(recipe_id: str):
    if not automation_service.delete_recipe(recipe_id):
        raise HTTPException(status_code=404, detail="Automation recipe not found.")
    return {"ok": True}


@app.get("/api/memory")
async def memory_items():
    return memory_service.list_items()


@app.post("/api/memory")
async def create_memory(request: AgentMemoryCreateRequest):
    return memory_service.create(request)


@app.put("/api/memory/{item_id}")
async def update_memory(item_id: str, request: AgentMemoryUpdateRequest):
    item = memory_service.update(item_id, request)
    if not item:
        raise HTTPException(status_code=404, detail="Memory item not found.")
    return item


@app.delete("/api/memory/{item_id}")
async def delete_memory(item_id: str):
    deleted = memory_service.delete(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory item not found.")
    return {"ok": True}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        return await document_service.upload_pdf(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/documents")
async def documents():
    return document_service.list_documents()


@app.post("/api/documents/{document_id}/ask")
async def ask_document(document_id: str, request: DocumentQuestionRequest):
    try:
        return await document_service.answer_question(document_id, request.question, agent_system.llm)
    except ValueError as exc:
        raise HTTPException(status_code=404 if "not found" in str(exc).lower() else 400, detail=str(exc)) from exc


@app.post("/api/study/generate")
async def generate_study_artifact(request: StudyGenerateRequest):
    artifact, setup = await study_service.generate(request)
    return {"artifact": artifact, "setup_required": setup}


@app.get("/api/study/artifacts")
async def study_artifacts():
    return study_service.list_artifacts()


@app.post("/api/mock-tests/generate")
async def generate_mock_test(request: MockTestGenerateRequest):
    try:
        test, setup = await mock_test_service.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"test": test, "setup_required": setup}


@app.get("/api/mock-tests")
async def mock_tests():
    return mock_test_service.list_tests()


@app.get("/api/mock-tests/{test_id}")
async def mock_test(test_id: str):
    test = mock_test_service.get_test(test_id)
    if not test:
        raise HTTPException(status_code=404, detail="Mock test not found.")
    return test


@app.post("/api/mock-tests/{test_id}/start")
async def start_mock_test(test_id: str):
    try:
        test, attempt = mock_test_service.start_attempt(test_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"test": test, "attempt": attempt}


@app.post("/api/mock-tests/{test_id}/attempts/{attempt_id}/submit")
async def submit_mock_test(test_id: str, attempt_id: str, request: MockTestSubmitRequest):
    try:
        return mock_test_service.submit_attempt(test_id, attempt_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/reports")
async def reports():
    return report_service.list_reports()


@app.get("/api/reports/{report_id}/download")
async def download_report(report_id: str):
    path = report_service.get_report_path(report_id)
    if not path:
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(path, media_type="text/markdown", filename=path.name)


@app.post("/api/voice/transcribe")
async def transcribe(audio: UploadFile | None = File(default=None)):
    filename = audio.filename if audio else "no file"
    return VoiceTranscriptionResponse(
        transcript="",
        message=(
            f"Received {filename}. Browser speech recognition is used for v1 hands-free voice; "
            "backend Whisper transcription can be plugged in later."
        ),
        setup_required=[],
    )


@app.get("/api/voice/status")
async def voice_status():
    return voice_service.status()


@app.post("/api/voice/warmup")
async def voice_warmup(request: VoiceWarmupRequest | None = None):
    status = voice_service.status()
    if not status.enabled:
        raise HTTPException(status_code=503, detail=status.model_dump())
    try:
        return await voice_service.warm_up(request.voice if request else None)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "enabled": False,
                "provider": "piper",
                "voice": request.voice if request and request.voice else settings.piper_voice_id,
                "cached": False,
                "loaded": False,
                "setup_required": ["PIPER_LOCAL_TTS"],
                "message": str(exc),
            },
        ) from exc


@app.post("/api/voice/speak")
async def speak(request: VoiceSpeakRequest):
    status = voice_service.status()
    if not status.enabled:
        raise HTTPException(status_code=503, detail=status.model_dump())

    try:
        audio, voice_id, cached = await voice_service.synthesize(request.text, request.voice)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "enabled": False,
                "provider": "piper",
                "voice": request.voice or settings.piper_voice_id,
                "cached": False,
                "loaded": False,
                "setup_required": ["PIPER_LOCAL_TTS"],
                "message": str(exc),
            },
        ) from exc

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "Cache-Control": "public, max-age=86400",
            "X-Astra-Voice-Provider": "piper",
            "X-Astra-Voice-Id": voice_id,
            "X-Astra-Voice-Cache": "hit" if cached else "miss",
        },
    )


@app.websocket("/ws/agent")
async def agent_socket(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"agent": "Astra", "status": "connected", "message": "Agent stream online."})
    try:
        while True:
            payload = await websocket.receive_json()
            message = payload.get("message", "")
            await websocket.send_json({"agent": "Supervisor", "status": "working", "message": f"Received: {message}"})
            await websocket.send_json({"agent": "Astra", "status": "complete", "message": "Use /api/chat or /api/research for full responses."})
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()
