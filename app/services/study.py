import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.models import StudyArtifact, StudyGenerateRequest
from app.services.documents import DocumentService
from app.services.llm import LlmService
from app.services.reports import ReportService


class StudyService:
    def __init__(self, settings: Settings, reports: ReportService, documents: DocumentService, llm: LlmService):
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.study_dir = base / "study"
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.study_dir / "index.json"
        self.reports = reports
        self.documents = documents
        self.llm = llm

    def list_artifacts(self) -> list[StudyArtifact]:
        return sorted(self._read_index(), key=lambda item: item.created_at, reverse=True)

    async def generate(self, request: StudyGenerateRequest) -> tuple[StudyArtifact, list[str]]:
        source_text = self._source_text(request)
        system_prompt = (
            "You are Astra's study agent. Create practical college study material. "
            "Use clear headings, compact bullets, and exam-friendly language."
        )
        user_prompt = (
            f"Artifact type: {request.artifact_type}\n"
            f"Topic: {request.topic}\n"
            f"Source material:\n{source_text[:6500] or request.topic}\n\n"
            "Return Markdown only."
        )
        markdown, setup = await self.llm.complete(system_prompt, user_prompt)
        artifact_id = uuid.uuid4().hex
        title = self._title(request)
        artifact = StudyArtifact(
            id=artifact_id,
            artifact_type=request.artifact_type,
            title=title,
            source=self._source_label(request),
            markdown=markdown,
            created_at=datetime.utcnow(),
        )
        (self.study_dir / f"{artifact_id}.md").write_text(markdown, encoding="utf-8")
        records = self._read_index()
        records.append(artifact)
        self._write_index(records)
        return artifact, setup

    def _source_text(self, request: StudyGenerateRequest) -> str:
        if request.source_text.strip():
            return request.source_text.strip()
        if request.report_id:
            path = self.reports.get_report_path(request.report_id)
            if path and path.exists():
                return path.read_text(encoding="utf-8")[:7000]
        if request.document_id:
            return self.documents.document_excerpt(request.document_id, max_chars=7000)
        return request.topic

    def _source_label(self, request: StudyGenerateRequest) -> str:
        if request.report_id:
            return f"report:{request.report_id}"
        if request.document_id:
            return f"document:{request.document_id}"
        if request.source_text.strip():
            return "custom text"
        return "topic"

    def _title(self, request: StudyGenerateRequest) -> str:
        label = request.artifact_type.replace("_", " ").title()
        topic = re.sub(r"\s+", " ", request.topic).strip()
        return f"{label}: {topic[:72]}"

    def _read_index(self) -> list[StudyArtifact]:
        if not self.index_path.exists():
            return []
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [StudyArtifact.model_validate(item) for item in raw if isinstance(item, dict)]

    def _write_index(self, artifacts: list[StudyArtifact]) -> None:
        payload = [artifact.model_dump(mode="json") for artifact in artifacts]
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
