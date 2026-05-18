import asyncio

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models import (
    AgentCommandRequest,
    AgentCommandTestRequest,
    AgentMemoryCreateRequest,
    AgentMemoryUpdateRequest,
    ChatRequest,
    CommandRequest,
    DocumentQuestionRequest,
    ResearchRequest,
    StudyGenerateRequest,
    VoiceSpeakRequest,
    VoiceWarmupRequest,
    VoiceTranscriptionResponse,
)
from app.services.agents import AstraAgentSystem
from app.services.commands import CommandService
from app.services.desktop import DesktopActionService
from app.services.documents import DocumentService
from app.services.memory import MemoryService
from app.services.reports import ReportService
from app.services.safe_agent import SafeAgentService
from app.services.study import StudyService
from app.services.voice import VoiceService

settings = get_settings()
agent_system = AstraAgentSystem(settings)
voice_service = VoiceService(settings)
report_service = ReportService(settings)
desktop_service = DesktopActionService()
memory_service = MemoryService(settings)
document_service = DocumentService(settings)
study_service = StudyService(settings, report_service, document_service, agent_system.llm)
safe_agent_service = SafeAgentService(settings, report_service, memory_service, document_service, study_service, desktop_service, voice_service)
command_service = CommandService(agent_system, desktop_service, report_service, safe_agent_service)

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
    return {
        "status": "ok",
        "app": "Astra AI Agent API",
        "model": settings.cerebras_model,
        "providers": {
            "cerebras": settings.has_cerebras,
            "tavily": settings.has_tavily,
            "openalex": True,
            "semantic_scholar": True,
            "duckduckgo_fallback": True,
            "piper_local": voice_service.status().enabled,
        },
    }


@app.get("/api/agents")
async def agents():
    return agent_system.descriptors()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return await agent_system.chat(request.message, request.mode)


@app.post("/api/research")
async def research(request: ResearchRequest):
    return await agent_system.research(request)


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
