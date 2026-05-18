import json
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings
from app.models import DocumentQuestionResponse, DocumentRecord
from app.services.llm import LlmService


class DocumentService:
    def __init__(self, settings: Settings):
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.documents_dir = base / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.documents_dir / "index.json"

    def list_documents(self) -> list[DocumentRecord]:
        return sorted(self._read_index(), key=lambda item: item.created_at, reverse=True)

    async def upload_pdf(self, upload: UploadFile) -> DocumentRecord:
        filename = upload.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF uploads are supported.")

        payload = await upload.read()
        if not payload:
            raise ValueError("The uploaded PDF is empty.")

        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(payload))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise ValueError(f"Astra could not read that PDF: {exc}") from exc

        clean_pages = [self._clean_text(page) for page in pages]
        combined = "\n".join(clean_pages).strip()
        if not combined:
            raise ValueError("No selectable text was found in this PDF. OCR support can be added later.")

        document_id = uuid.uuid4().hex
        document_dir = self.documents_dir / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        (document_dir / "original.pdf").write_bytes(payload)
        (document_dir / "pages.json").write_text(json.dumps(clean_pages, indent=2), encoding="utf-8")

        title = Path(filename).stem.replace("-", " ").replace("_", " ").strip() or "Uploaded PDF"
        record = DocumentRecord(
            id=document_id,
            title=title,
            filename=filename,
            created_at=datetime.utcnow(),
            page_count=len(clean_pages),
            text_preview=combined[:260],
        )
        records = self._read_index()
        records.append(record)
        self._write_index(records)
        return record

    async def answer_question(self, document_id: str, question: str, llm: LlmService) -> DocumentQuestionResponse:
        record = self.get_document(document_id)
        if not record:
            raise ValueError("Document not found.")

        pages = self._read_pages(document_id)
        if not pages:
            raise ValueError("Document text is not available.")

        selected = self._select_pages(pages, question)
        evidence = "\n\n".join(f"Page {page_number}:\n{text[:1800]}" for page_number, text in selected)
        system_prompt = (
            "You are Astra's document agent. Answer using only the provided PDF page text. "
            "Mention page numbers when evidence supports the answer."
        )
        user_prompt = f"Question: {question}\n\nPDF evidence:\n{evidence}\n\nAnswer with concise page references."
        answer, setup = await llm.complete(system_prompt, user_prompt)
        return DocumentQuestionResponse(
            document=record,
            answer=answer,
            page_refs=[page_number for page_number, _ in selected],
            setup_required=setup,
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        safe_id = self._safe_id(document_id)
        if not safe_id:
            return None
        for record in self._read_index():
            if record.id == safe_id:
                return record
        return None

    def document_excerpt(self, document_id: str, max_chars: int = 5000) -> str:
        pages = self._read_pages(document_id)
        combined = "\n\n".join(f"Page {index + 1}: {text}" for index, text in enumerate(pages))
        return combined[:max_chars]

    def _read_index(self) -> list[DocumentRecord]:
        if not self.index_path.exists():
            return []
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [DocumentRecord.model_validate(item) for item in raw if isinstance(item, dict)]

    def _write_index(self, records: list[DocumentRecord]) -> None:
        payload = [record.model_dump(mode="json") for record in records]
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_pages(self, document_id: str) -> list[str]:
        safe_id = self._safe_id(document_id)
        if not safe_id:
            return []
        path = self.documents_dir / safe_id / "pages.json"
        if not path.exists() or path.parent.parent.resolve() != self.documents_dir.resolve():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [str(item) for item in raw if isinstance(item, str)]

    def _select_pages(self, pages: list[str], question: str) -> list[tuple[int, str]]:
        terms = {term for term in re.findall(r"[a-zA-Z0-9]{3,}", question.lower())}
        scored: list[tuple[int, int, str]] = []
        for index, text in enumerate(pages, start=1):
            lowered = text.lower()
            score = sum(lowered.count(term) for term in terms)
            scored.append((score, index, text))
        chosen = sorted(scored, key=lambda item: (item[0], -item[1]), reverse=True)[:4]
        if not any(score for score, _, _ in chosen):
            chosen = [(0, index + 1, text) for index, text in enumerate(pages[:4])]
        return [(index, text) for _, index, text in sorted(chosen, key=lambda item: item[1])]

    def _safe_id(self, value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "", value).strip()

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
