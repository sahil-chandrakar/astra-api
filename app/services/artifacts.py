from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import AutomationArtifact, AutomationArtifactMediaType


MEDIA_EXTENSIONS: dict[str, AutomationArtifactMediaType] = {
    ".mp4": "video",
    ".mkv": "video",
    ".webm": "video",
    ".mov": "video",
    ".avi": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".gif": "image",
    ".pdf": "document",
    ".doc": "document",
    ".docx": "document",
    ".md": "text",
    ".txt": "text",
    ".csv": "text",
    ".json": "text",
    ".zip": "archive",
    ".7z": "archive",
    ".rar": "archive",
}


class ArtifactService:
    """Persistent local memory for files Astra created or downloaded."""

    def __init__(self, settings: Settings):
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.automation_dir = base / "automation"
        self.downloads_dir = self.automation_dir / "downloads"
        self.artifact_path = self.automation_dir / "artifacts.json"
        self.automation_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    def list_recent(self, media_types: list[str] | None = None, limit: int = 25) -> list[AutomationArtifact]:
        allowed = {item for item in media_types or [] if item}
        artifacts = [item for item in self._read() if self._artifact_exists(item)]
        if allowed:
            artifacts = [item for item in artifacts if item.media_type in allowed]
        return sorted(artifacts, key=lambda item: item.created_at, reverse=True)[:limit]

    def get(self, artifact_id: str) -> AutomationArtifact | None:
        return next((item for item in self._read() if item.id == artifact_id and self._artifact_exists(item)), None)

    def get_by_path(self, path: str | Path) -> AutomationArtifact | None:
        try:
            target = Path(path).expanduser().resolve()
        except Exception:
            return None
        return next((item for item in self._read() if self._safe_resolve(item.path) == target and self._artifact_exists(item)), None)

    def record_download(
        self,
        path: str | Path,
        run_id: str,
        source_url: str = "",
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AutomationArtifact:
        target = Path(path).expanduser().resolve()
        self._ensure_owned_path(target)
        now = datetime.utcnow()
        existing = self.get_by_path(target)
        artifact = AutomationArtifact(
            id=existing.id if existing else uuid.uuid4().hex,
            kind="download",
            media_type=self.media_type_for_path(target),
            title=title.strip() or target.stem,
            filename=target.name,
            path=str(target),
            source_url=source_url.strip(),
            run_id=run_id,
            size_bytes=target.stat().st_size if target.exists() else 0,
            metadata=metadata or {},
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        items = [item for item in self._read() if item.id != artifact.id]
        items.append(artifact)
        self._write(items)
        return artifact

    def resolve_reference(
        self,
        query: str,
        media_types: list[str] | None = None,
        limit: int = 5,
        preferred_artifact_id: str = "",
    ) -> tuple[AutomationArtifact | None, list[AutomationArtifact], str]:
        inferred = media_types or self.infer_media_types(query)
        candidates = self.list_recent(inferred, limit=50)
        if not candidates:
            return None, [], "No matching Astra artifact was found."

        normalized_query = self._normalize_reference(query)
        preferred = next((item for item in candidates if item.id == preferred_artifact_id), None)
        if preferred and self._is_latest_reference(normalized_query):
            return preferred, [preferred, *[item for item in candidates if item.id != preferred.id]][:limit], ""
        if not normalized_query or self._is_latest_reference(normalized_query):
            return candidates[0], candidates[:limit], ""

        scored = sorted(
            ((self._artifact_score(normalized_query, artifact), artifact) for artifact in candidates),
            key=lambda item: (item[0], item[1].created_at),
            reverse=True,
        )
        best_score, best_artifact = scored[0]
        matches = [artifact for score, artifact in scored if score >= max(42, best_score - 8)][:limit]
        if best_score < 42:
            return None, candidates[:limit], "No artifact matched that description."
        if len(matches) > 1 and (best_score < 86 or scored[1][0] == best_score):
            return None, matches, "Multiple artifacts matched that description."
        return best_artifact, matches, ""

    def infer_media_types(self, query: str) -> list[str]:
        normalized = query.lower()
        if re.search(r"\b(song|audio|music|track|mp3|wav|listen)\b", normalized):
            return ["audio", "video"]
        if re.search(r"\b(video|movie|clip|mp4|mkv|watch|play)\b", normalized):
            return ["video", "audio"]
        if re.search(r"\b(pdf|document|doc|report|paper)\b", normalized):
            return ["document", "text"]
        if re.search(r"\b(image|photo|picture|screenshot)\b", normalized):
            return ["image"]
        return []

    def media_type_for_path(self, path: str | Path) -> AutomationArtifactMediaType:
        return MEDIA_EXTENSIONS.get(Path(path).suffix.lower(), "unknown")

    def ensure_artifact_path(self, artifact: AutomationArtifact) -> Path:
        target = self._safe_resolve(artifact.path)
        if not target or not target.exists() or not target.is_file():
            raise ValueError("The resolved artifact file is no longer available.")
        self._ensure_owned_path(target)
        return target

    def _artifact_score(self, normalized_query: str, artifact: AutomationArtifact) -> int:
        haystack = self._normalize_reference(" ".join([artifact.title, artifact.filename, artifact.source_url]))
        if not haystack:
            return 0
        if normalized_query in haystack:
            return 100
        query_tokens = set(normalized_query.split())
        haystack_tokens = set(haystack.split())
        overlap = len(query_tokens & haystack_tokens) * 18 if query_tokens else 0
        ratio = int(SequenceMatcher(None, normalized_query, haystack).ratio() * 100)
        partial = max((int(SequenceMatcher(None, token, haystack).ratio() * 100) for token in query_tokens), default=0)
        recency_bonus = 8
        return min(100, max(ratio, partial) + overlap + recency_bonus)

    def _normalize_reference(self, text: str) -> str:
        text = text.lower()
        text = re.sub(
            r"\b(that|the|my|a|an|is|are|was|be|please|downloaded|download|file|folder|directory|where|located|location|containing|"
            r"latest|recent|previous|last|open|show|reveal|find|play|view|watch|listen|for|in|with|using|vlc|default)\b",
            " ",
            text,
        )
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _is_latest_reference(self, normalized_query: str) -> bool:
        return normalized_query in {"", "song", "audio", "music", "video", "movie", "clip", "media"}

    def _artifact_exists(self, artifact: AutomationArtifact) -> bool:
        target = self._safe_resolve(artifact.path)
        return bool(target and target.exists() and target.is_file())

    def _safe_resolve(self, path: str) -> Path | None:
        try:
            return Path(path).expanduser().resolve()
        except Exception:
            return None

    def _ensure_owned_path(self, path: Path) -> None:
        downloads_root = self.downloads_dir.resolve()
        if path != downloads_root and downloads_root not in path.parents:
            raise ValueError("Astra can only use files inside its automation downloads folder.")

    def _read(self) -> list[AutomationArtifact]:
        if not self.artifact_path.exists():
            return []
        try:
            raw = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        artifacts: list[AutomationArtifact] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                artifacts.append(AutomationArtifact.model_validate(item))
            except Exception:
                continue
        return artifacts

    def _write(self, artifacts: list[AutomationArtifact]) -> None:
        payload = [item.model_dump(mode="json") for item in artifacts]
        self.artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
