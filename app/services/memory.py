import json
import uuid
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.models import AgentMemoryCreateRequest, AgentMemoryItem, AgentMemoryUpdateRequest


class MemoryService:
    def __init__(self, settings: Settings):
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.memory_dir = base / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.memory_dir / "memory.json"

    def list_items(self) -> list[AgentMemoryItem]:
        return sorted(self._read(), key=lambda item: item.updated_at, reverse=True)

    def create(self, request: AgentMemoryCreateRequest) -> AgentMemoryItem:
        now = datetime.utcnow()
        item = AgentMemoryItem(
            id=uuid.uuid4().hex,
            category=request.category,
            text=request.text.strip(),
            created_at=now,
            updated_at=now,
        )
        items = self._read()
        items.append(item)
        self._write(items)
        return item

    def update(self, item_id: str, request: AgentMemoryUpdateRequest) -> AgentMemoryItem | None:
        items = self._read()
        updated: AgentMemoryItem | None = None
        for index, item in enumerate(items):
            if item.id != item_id:
                continue
            updated = item.model_copy(
                update={
                    "category": request.category or item.category,
                    "text": request.text.strip() if request.text is not None else item.text,
                    "updated_at": datetime.utcnow(),
                }
            )
            items[index] = updated
            break
        if updated:
            self._write(items)
        return updated

    def delete(self, item_id: str) -> bool:
        items = self._read()
        remaining = [item for item in items if item.id != item_id]
        if len(remaining) == len(items):
            return False
        self._write(remaining)
        return True

    def _read(self) -> list[AgentMemoryItem]:
        if not self.memory_path.exists():
            return []
        try:
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [AgentMemoryItem.model_validate(item) for item in raw if isinstance(item, dict)]

    def _write(self, items: list[AgentMemoryItem]) -> None:
        payload = [item.model_dump(mode="json") for item in items]
        self.memory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
