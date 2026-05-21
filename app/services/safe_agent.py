from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback only used when optional dependency is missing
    fuzz = None

from app.config import Settings
from app.models import (
    AgentAbility,
    AgentAuditEntry,
    AgentCommandOutcome,
    AgentCommandRequest,
    AgentCommandResponse,
    AgentCommandRisk,
    AgentCommandTestResponse,
    AgentEvent,
    AgentMemoryCreateRequest,
    MockTestGenerateRequest,
    StudyGenerateRequest,
)
from app.services.agent_intent import AgentIntentParser
from app.services.desktop import SAFE_TARGETS, DesktopActionService, DesktopTarget
from app.services.documents import DocumentService
from app.services.memory import MemoryService
from app.services.mock_tests import MockTestService, MockTestSourceMaterialError
from app.services.reports import ReportService
from app.services.study import StudyService
from app.services.voice import VoiceService


Executor = Callable[[dict[str, Any]], Awaitable[tuple[bool, str, dict[str, Any]]]]
Tester = Callable[[], Awaitable[tuple[bool, str]]]


@dataclass(frozen=True)
class AgentCommandDefinition:
    id: str
    label: str
    description: str
    category: str
    risk: AgentCommandRisk
    params_schema: dict[str, Any] = field(default_factory=dict)
    required_params: tuple[str, ...] = ()
    executor: Executor | None = None
    tester: Tester | None = None


@dataclass(frozen=True)
class IntentCandidate:
    command_id: str
    label: str
    aliases: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IntentMatch:
    score: int
    candidate: IntentCandidate
    matched_alias: str
    second_score: int = 0
    candidate_scores: tuple[dict[str, Any], ...] = ()


class SafeAgentService:
    def __init__(
        self,
        settings: Settings,
        reports: ReportService,
        memory: MemoryService,
        documents: DocumentService,
        study: StudyService,
        mock_tests: MockTestService,
        desktop: DesktopActionService,
        voice: VoiceService,
    ):
        self.settings = settings
        self.reports = reports
        self.memory = memory
        self.documents = documents
        self.study = study
        self.mock_tests = mock_tests
        self.desktop = desktop
        self.voice = voice
        self.llm = study.llm
        self.intent_parser = AgentIntentParser(settings, self.llm)

        self.backend_dir = Path(__file__).resolve().parents[2]
        self.project_root = self.backend_dir.parent
        self.frontend_dir = self.project_root / "astra-frontend"

        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.agent_dir = base / "agent"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.agent_dir / "audit.jsonl"
        self.test_path = self.agent_dir / "command-tests.json"
        self.correction_path = self.agent_dir / "intent-corrections.json"
        self.safe_targets = self._safe_target_map()
        self.commands = self._build_commands()
        self.intent_candidates = self._build_intent_candidates()

    def abilities(self) -> list[AgentAbility]:
        test_results = self._read_test_results()
        return [self._ability_for(command, test_results.get(command.id, {})) for command in self.commands.values()]

    async def test_commands(self, command_id: str | None = None) -> AgentCommandTestResponse:
        targets = [self.commands[command_id]] if command_id and command_id in self.commands else list(self.commands.values())
        events: list[AgentEvent] = []
        for command in targets:
            passed, message = await self._run_self_test(command)
            events.append(
                AgentEvent(
                    agent="Safety Validator",
                    status="complete" if passed else "warning",
                    message=f"{command.label}: {message}",
                )
            )
        return AgentCommandTestResponse(abilities=self.abilities(), events=events)

    def audit_entries(self, limit: int = 50) -> list[AgentAuditEntry]:
        if not self.audit_path.exists():
            return []
        entries: list[AgentAuditEntry] = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(AgentAuditEntry.model_validate(json.loads(line)))
            except Exception:
                continue
        return list(reversed(entries))[:limit]

    async def handle_natural_language(self, text: str, confirmed: bool = False) -> AgentCommandResponse | None:
        request = await self.resolve_request_from_text(text, confirmed=confirmed)
        if not request:
            return None
        return await self.execute(request)

    def request_from_text(self, text: str, confirmed: bool = False) -> AgentCommandRequest | None:
        normalized = self._normalize(text)
        if not normalized:
            return None

        unsafe = self._unsafe_request(text, normalized, confirmed)
        if unsafe:
            return unsafe

        mock_follow_up = self._resolve_mock_test_follow_up(text, normalized, confirmed)
        if mock_follow_up:
            return mock_follow_up

        saved = self._resolve_saved_correction(text, normalized, confirmed)
        if saved:
            return saved

        local_drive = self._resolve_local_drive_request(text, normalized, confirmed)
        if local_drive:
            return local_drive

        exact = self._resolve_exact_candidate(text, normalized, confirmed)
        if exact:
            return exact

        legacy = self._legacy_request_from_text(text, confirmed=confirmed)
        if legacy:
            return legacy

        return self._resolve_fuzzy_candidate(text, normalized, confirmed)

    async def resolve_request_from_text(self, text: str, confirmed: bool = False) -> AgentCommandRequest | None:
        normalized = self._normalize(text)
        if not normalized:
            return None

        unsafe = self._unsafe_request(text, normalized, confirmed)
        if unsafe:
            return unsafe

        semantic = await self.intent_parser.resolve(text, normalized, confirmed, set(self.commands))
        if semantic:
            return semantic

        request = self.request_from_text(text, confirmed=confirmed)
        if request:
            return request
        return await self._resolve_with_llm(text, normalized, confirmed)

    def _unsafe_request(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        destructive = ("delete", "remove", "wipe", "erase", "rm", "rmdir", "format")
        control = ("click", "type", "move", "close", "control", "press", "shell", "terminal", "cmd", "powershell", "api key")
        memory_scoped = "memory" in normalized or self._has_fuzzy_word(normalized, ("memory",), threshold=84)
        if self._has_unsafe_word(normalized, destructive, threshold=82) and not memory_scoped:
            return AgentCommandRequest(
                command_id="blocked_action",
                input_text=text,
                params={"reason": "destructive file/system request"},
                confirmed=confirmed,
                resolution=self._resolution_meta("safety", 1.0, "destructive", normalized),
            )
        if self._has_unsafe_word(normalized, control, threshold=82):
            return AgentCommandRequest(
                command_id="blocked_action",
                input_text=text,
                params={"reason": "free-form desktop or shell control"},
                confirmed=confirmed,
                resolution=self._resolution_meta("safety", 1.0, "desktop/shell control", normalized),
            )
        return None

    def _resolve_mock_test_follow_up(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        if not self._mentions_mock_test(normalized):
            return None
        if self._is_mock_generation_intent(normalized):
            return None
        if not self._is_mock_existing_intent(normalized):
            return None

        topic = self._extract_existing_mock_topic(text)
        resolution = self._resolution_meta("intent", 1.0, "mock test follow-up", normalized)
        resolution["intent"] = "open_existing_tool"
        if topic:
            resolution["topic"] = topic
        return AgentCommandRequest(
            command_id="open_latest_mock_test",
            input_text=text,
            params={"topic": topic} if topic else {},
            confirmed=confirmed,
            resolution=resolution,
        )

    def _resolve_exact_candidate(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        best: tuple[int, IntentCandidate, str] | None = None
        for candidate in self.intent_candidates:
            if candidate.command_id == "open_allowlisted_target" and not self._open_candidate_allowed(normalized):
                continue
            if candidate.command_id == "generate_mock_test" and not self._is_mock_generation_intent(normalized):
                continue
            for alias in candidate.aliases:
                alias_normalized = self._normalize(alias)
                if alias_normalized and re.search(rf"\b{re.escape(alias_normalized)}\b", normalized):
                    specificity = len(self._compact(alias_normalized))
                    if best is None or specificity > best[0]:
                        best = (specificity, candidate, alias_normalized)
        if not best:
            return None
        _, candidate, alias_normalized = best
        return self._request_for_candidate(candidate, text, normalized, "exact", 1.0, alias_normalized, confirmed)

    def _resolve_fuzzy_candidate(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        best = self._best_intent_match(normalized)
        if not best or best.score < 72:
            return None

        command = self.commands.get(best.candidate.command_id)
        high_confidence = best.score >= 92
        close_second = best.second_score >= 72 and best.score - best.second_score <= 6
        needs_confirmation = (not high_confidence) or close_second
        if command and command.risk != "safe_auto":
            needs_confirmation = True
        return self._request_for_candidate(
            best.candidate,
            text,
            normalized,
            "fuzzy",
            best.score / 100,
            best.matched_alias,
            confirmed,
            needs_confirmation=needs_confirmation,
            second_score=best.second_score / 100 if best.second_score else None,
            candidate_scores=best.candidate_scores,
        )

    async def _resolve_with_llm(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        if not self.settings.has_cerebras:
            return None

        candidates = self._llm_candidates()
        system_prompt = (
            "You resolve a user's agent prompt to one existing safe command. "
            "First classify whether the user wants a new command, a status check, a complaint, a follow-up, or chat. "
            "For mock-test status/complaint prompts like 'where is my mock test' or 'mock not created', choose open_latest_mock_test, not generate_mock_test. "
            "Choose generate_mock_test only when the user clearly asks to create, generate, make, build, or prepare a new test. "
            "Return strict JSON only. Do not invent command IDs or executable strings. "
            "If no candidate fits, return command_id null."
        )
        user_prompt = json.dumps(
            {
                "prompt": text,
                "normalized_prompt": normalized,
                "allowed_candidates": candidates,
                "output_schema": {
                    "command_id": "string|null",
                    "params": "object",
                    "confidence": "0.0-1.0",
                    "matched_alias": "string",
                    "reason": "short string",
                },
            },
            ensure_ascii=True,
        )
        raw, setup = await self.llm.complete(system_prompt, user_prompt)
        if setup:
            return None
        payload = self._parse_llm_json(raw)
        if not payload:
            return None

        command_id = payload.get("command_id")
        if not command_id:
            return None

        try:
            confidence = float(payload.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        if confidence < 0.72:
            return None

        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        matched_alias = str(payload.get("matched_alias") or payload.get("reason") or "llm")
        resolution = self._resolution_meta("llm", confidence, matched_alias, normalized)
        resolution["reason"] = str(payload.get("reason") or "")[:180]

        if command_id == "generate_mock_test" and not self._is_mock_generation_intent(normalized):
            if self._is_mock_existing_intent(normalized):
                topic = self._extract_existing_mock_topic(text)
                resolution["intent"] = "open_existing_tool"
                return AgentCommandRequest(
                    command_id="open_latest_mock_test",
                    input_text=text,
                    params={"topic": topic} if topic else {},
                    confirmed=confirmed,
                    resolution=resolution,
                )
            return None

        if command_id == "clarify_agent_intent" and not re.search(
            r"\b(pyq|previous\s+year|past\s+paper|official\s+questions?|year\s+questions?)\b",
            normalized,
        ):
            return None

        if command_id not in self.commands:
            return AgentCommandRequest(command_id=str(command_id), input_text=text, params=params, confirmed=confirmed, resolution=resolution)

        command = self.commands[str(command_id)]
        if confidence < 0.92 or command.risk != "safe_auto":
            resolution["needs_confirmation"] = True

        if command_id == "open_arbitrary_url" and not self._extract_direct_url(text):
            return AgentCommandRequest(
                command_id="blocked_action",
                input_text=text,
                params={"reason": "LLM suggested opening an arbitrary URL without a typed http(s) URL"},
                confirmed=confirmed,
                resolution=resolution,
            )

        return AgentCommandRequest(command_id=str(command_id), input_text=text, params=params, confirmed=confirmed, resolution=resolution)

    def _resolve_saved_correction(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        correction = self._read_intent_corrections().get(normalized)
        if not isinstance(correction, dict):
            return None

        command_id = str(correction.get("command_id") or "")
        command = self.commands.get(command_id)
        params = correction.get("params") if isinstance(correction.get("params"), dict) else {}
        if not command:
            return None

        clean_params, validation_error = self._validate_params(command, dict(params))
        if validation_error:
            return None

        confidence = correction.get("confidence")
        try:
            saved_confidence = float(confidence)
        except (TypeError, ValueError):
            saved_confidence = 1.0
        resolution = self._resolution_meta(
            "correction",
            saved_confidence,
            str(correction.get("matched_alias") or "confirmed correction"),
            normalized,
        )
        resolution["correction_source"] = "confirmed"
        resolution["saved_at"] = str(correction.get("updated_at") or correction.get("created_at") or "")
        self._mark_intent_correction_used(normalized, correction)
        return AgentCommandRequest(command_id=command_id, input_text=text, params=clean_params, confirmed=confirmed, resolution=resolution)

    def _resolve_local_drive_request(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        drive = self._extract_local_drive_letter(normalized)
        if not drive:
            return None
        if not self._open_candidate_allowed(normalized):
            return None
        explorer_requested = self._has_fuzzy_word(normalized, ("file explorer", "explorer", "files", "folder"), threshold=82)
        local_drive_requested = bool(re.search(rf"\b{drive.lower()}\s*(?::|drive|disk)\b|\b(?:drive|disk)\s+{drive.lower()}\b", normalized))
        if not explorer_requested and not local_drive_requested:
            return None
        resolution = self._resolution_meta("exact", 1.0, f"{drive}: drive", normalized)
        resolution["local_drive"] = drive
        return AgentCommandRequest(
            command_id="open_allowlisted_target",
            input_text=text,
            params={"target": "file_explorer", "drive": drive},
            confirmed=confirmed,
            resolution=resolution,
        )

    def _best_intent_match(self, normalized: str) -> IntentMatch | None:
        scores: list[tuple[int, IntentCandidate, str]] = []
        for candidate in self.intent_candidates:
            if candidate.command_id == "open_allowlisted_target" and not self._open_candidate_allowed(normalized):
                continue
            score, matched_alias = self._score_intent_candidate(normalized, candidate)
            if score > 0:
                scores.append((score, candidate, matched_alias))

        if not scores:
            return None

        scores.sort(key=lambda item: item[0], reverse=True)
        top_score, top_candidate, top_alias = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0
        candidate_scores = tuple(
            {
                "command_id": candidate.command_id,
                "label": candidate.label,
                "params": candidate.params,
                "matched_alias": alias,
                "confidence": round(score / 100, 3),
            }
            for score, candidate, alias in scores[:5]
        )
        return IntentMatch(
            score=top_score,
            candidate=top_candidate,
            matched_alias=top_alias,
            second_score=second_score,
            candidate_scores=candidate_scores,
        )

    def _score_intent_candidate(self, normalized: str, candidate: IntentCandidate) -> tuple[int, str]:
        best = (0, "")
        if candidate.command_id == "open_allowlisted_target":
            return self._open_target_score(normalized, str(candidate.params.get("target", "")))
        if candidate.command_id == "generate_mock_test" and not self._is_mock_generation_intent(normalized):
            return 0, ""

        for alias in candidate.aliases:
            alias_normalized = self._normalize(alias)
            if not alias_normalized:
                continue
            if any(len(part) <= 3 for part in alias_normalized.split()) and candidate.command_id != "open_allowlisted_target":
                continue
            score = self._alias_score(normalized, alias_normalized)
            if score > best[0]:
                best = (score, alias_normalized)
        return best

    def _request_for_candidate(
        self,
        candidate: IntentCandidate,
        text: str,
        normalized: str,
        source: str,
        confidence: float,
        matched_alias: str,
        confirmed: bool,
        needs_confirmation: bool = False,
        second_score: float | None = None,
        candidate_scores: tuple[dict[str, Any], ...] = (),
    ) -> AgentCommandRequest | None:
        params = dict(candidate.params)
        if candidate.command_id == "save_memory":
            memory_text = self._extract_memory_text(text, matched_alias)
            if not memory_text:
                return None
            params = {"category": self._infer_memory_category(memory_text), "text": memory_text}
        elif candidate.command_id == "generate_study_artifact":
            topic = self._extract_study_topic(text, matched_alias)
            params = {**params, "topic": topic or text.strip()}
        elif candidate.command_id == "generate_mock_test":
            topic = self._extract_mock_test_topic(text, matched_alias)
            params = {
                "topic": topic or text.strip(),
                "question_count": int(params.get("question_count", 10)),
                "difficulty": str(params.get("difficulty", "mixed")),
                "mode": "mcq",
                "duration_minutes": int(params.get("duration_minutes", 20)),
            }

        resolution = self._resolution_meta(source, confidence, matched_alias, normalized)
        if needs_confirmation:
            resolution["needs_confirmation"] = True
        if second_score is not None:
            resolution["second_best"] = round(max(0.0, min(1.0, second_score)), 3)
        if candidate_scores:
            resolution["candidate_scores"] = list(candidate_scores[:5])
        return AgentCommandRequest(command_id=candidate.command_id, input_text=text, params=params, confirmed=confirmed, resolution=resolution)

    def _open_candidate_allowed(self, normalized: str) -> bool:
        tokens = normalized.split()
        if self._looks_like_question(normalized):
            return False
        return self._is_open_intent(normalized) or len(tokens) <= 3

    def _open_target_score(self, normalized: str, target_key: str) -> tuple[int, str]:
        target = self.safe_targets.get(target_key)
        if not target:
            return 0, ""
        target_text = self._target_text(normalized)
        aliases = {target.label, *target.aliases}
        if target_key == "file_explorer":
            aliases.update({"file manager", "windows explorer", "folders", "folder", "my files"})
        best = (0, "")
        target_word_count = len(target_text.split())
        for alias in aliases:
            alias_normalized = self._normalize(alias)
            if target_word_count == 1 and len(alias_normalized.split()) > 1:
                continue
            score = self._alias_score(target_text, alias_normalized)
            if score > best[0]:
                best = (score, alias_normalized)
        return best

    def _target_text(self, normalized: str) -> str:
        words = normalized.split()
        filtered = [word for word in words if self._score(word, "open") < 70 and word not in {"launch", "start", "go", "to", "visit"}]
        return " ".join(filtered).strip() or normalized

    def _extract_local_drive_letter(self, normalized: str) -> str | None:
        patterns = (
            r"\b(?P<drive>[a-z])\s*:(?=\s|$)",
            r"\b(?P<drive>[a-z])\s+(?:drive|disk)\b",
            r"\b(?:drive|disk)\s+(?P<drive>[a-z])\b",
            r"\blocal\s+(?:drive|disk)\s+(?P<drive>[a-z])\b",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group("drive").upper()
        return None

    def _is_open_intent(self, normalized: str) -> bool:
        if re.search(r"\b(open|launch|start|go to|visit)\b", normalized):
            return True
        return self._has_fuzzy_word(normalized, ("open", "launch", "start", "visit"), threshold=80)

    def _looks_like_question(self, normalized: str) -> bool:
        return bool(re.search(r"\b(what|why|how|when|where|who|explain|tell me|search for)\b", normalized))

    def _mentions_mock_test(self, normalized: str) -> bool:
        compact = self._compact(normalized)
        if "mocktest" in compact or "mockexam" in compact or "practicetest" in compact:
            return True
        if re.search(r"\b(mock|exam|practice)\b", normalized):
            return True
        if self._is_assessment_generation_intent(normalized):
            return True
        if re.search(r"\btest\b", normalized) and re.search(r"\b(create|generate|make|build|prepare|new|another|creat|genrate|mak)\b", normalized):
            return not re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized)
        return False

    def _is_mock_generation_intent(self, normalized: str) -> bool:
        if not self._mentions_mock_test(normalized):
            return False
        if re.search(r"\b(not created|not generated|did not create|didnt create|not showing|till now|where|show|open|view|see|find)\b", normalized):
            return False
        if self._is_assessment_generation_intent(normalized):
            return True
        return bool(re.search(r"\b(create|generate|make|build|prepare|new|another|creat|genrate|generte|mak|preprare)\b", normalized))

    def _is_assessment_generation_intent(self, normalized: str) -> bool:
        if re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized):
            return False
        if re.search(r"\b(test|quiz|assess|challenge)\s+(me|my knowledge)?\b", normalized):
            return True
        if re.search(r"\b(ask|give)\s+(me\s+)?(some\s+|a\s+)?(questions?|mcqs?|quiz)\b", normalized):
            return True
        return False

    def _is_mock_existing_intent(self, normalized: str) -> bool:
        if not self._mentions_mock_test(normalized):
            return False
        status_markers = (
            "where",
            "show",
            "open",
            "view",
            "see",
            "find",
            "latest",
            "created",
            "generated",
            "available",
            "missing",
            "not created",
            "not generated",
            "did not create",
            "didnt create",
            "not showing",
            "till now",
            "yet",
            "done",
            "ready",
        )
        return any(marker in normalized for marker in status_markers)

    def _mock_topic_looks_like_status(self, topic: str) -> bool:
        normalized = self._normalize(topic)
        return bool(re.search(r"\b(not created|not generated|did not create|didnt create|not showing|till now|where|show|open|view|see|find)\b", normalized))

    def _extract_memory_text(self, text: str, matched_alias: str) -> str:
        cleaned = re.sub(r"^\s*(please\s+)?(remember|rember|remeber|save memory|add memory|memorize|memo)\b\s*", "", text.strip(), flags=re.IGNORECASE)
        if cleaned == text.strip() and matched_alias:
            cleaned = re.sub(rf"^\s*(please\s+)?{re.escape(matched_alias)}\b\s*", "", text.strip(), flags=re.IGNORECASE)
        return cleaned.strip(" .")

    def _extract_study_topic(self, text: str, matched_alias: str) -> str:
        cleaned = re.sub(
            r"^\s*(please\s+)?(generate|create|make|build|genrate|creat|mak)\s+(me\s+)?(study\s+)?(notes|flashcards|flashcrds|quiz|quizz|revision plan|viva questions)\s*(for|on|about)?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip(" .")
        if cleaned == text.strip() and matched_alias:
            cleaned = re.sub(rf"^\s*(please\s+)?{re.escape(matched_alias)}\s*(for|on|about)?\s*", "", text.strip(), flags=re.IGNORECASE).strip(" .")
        return cleaned

    def _extract_mock_test_topic(self, text: str, matched_alias: str) -> str:
        cleaned = re.sub(
            r"^\s*(please\s+)?(generate|create|make|build|prepare|genrate|creat|mak)\s+(me\s+)?(a\s+|another\s+|new\s+)?(mock\s+test|practice\s+test|mock\s+exam|test|exam)\s*(for|on|about|of)?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip(" .")
        cleaned = re.sub(
            r"^\s*(please\s+)?(test|quiz|assess|challenge)\s+(me|my knowledge)?\s*(on|about|in|for)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .")
        cleaned = re.sub(
            r"^\s*(please\s+)?(ask|give)\s+(me\s+)?(some\s+|a\s+)?(questions?|mcqs?|quiz|test)\s*(on|about|in|for)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .")
        cleaned = re.sub(
            r"^\s*(please\s+)?(make|create|generate|build|prepare|genrate|creat|mak)\s+(another\s+|new\s+)?(?P<topic>.+?)\s+(mock\s+test|practice\s+test|mock\s+exam|test|exam)\s*$",
            lambda match: match.group("topic"),
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" .")
        cleaned = re.sub(r"\b(current affairs?)\s+(on|about|for)\s+(the\s+)?topic\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(current affairs?)\s+(on|about|for)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(on|about|for)\s+(the\s+)?topic\s+", "", cleaned, flags=re.IGNORECASE)
        if cleaned == text.strip() and matched_alias:
            cleaned = re.sub(rf"^\s*(please\s+)?{re.escape(matched_alias)}\s*(for|on|about|of)?\s*", "", text.strip(), flags=re.IGNORECASE).strip(" .")
        return cleaned

    def _extract_existing_mock_topic(self, text: str) -> str:
        cleaned = self._normalize(text)
        cleaned = re.sub(r"\b(where|is|are|my|the|latest|show|open|view|see|find|please|mock|test|exam|practice|created|generated|available|missing|not|did|didnt|create|showing|till|now|yet|done|ready)\b", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        return cleaned

    def _has_fuzzy_word(self, normalized: str, aliases: tuple[str, ...], threshold: int) -> bool:
        variants = self._match_variants(normalized)
        words = [word for variant in variants for word in variant.split()]
        if not words:
            return False
        for alias in aliases:
            alias_normalized = self._normalize(alias)
            alias_variants = self._match_variants(alias_normalized)
            if " " in alias_normalized:
                if any(alias_variant in variant for variant in variants for alias_variant in alias_variants):
                    return True
                if any(self._alias_score(window, alias_normalized) >= threshold for window in self._token_windows(words, len(alias_normalized.split()))):
                    return True
                if self._alias_score(normalized, alias_normalized) >= threshold:
                    return True
            if alias_normalized in words:
                return True
            if len(alias_normalized) <= 3:
                continue
            if any(len(word) > 3 and self._alias_score(word, alias_normalized) >= threshold for word in words):
                return True
        return False

    def _has_unsafe_word(self, normalized: str, aliases: tuple[str, ...], threshold: int) -> bool:
        variants = tuple(
            dict.fromkeys(
                variant
                for variant in (
                    normalized,
                    self._squash_repeats(normalized, 2),
                    self._squash_repeats(normalized, 1),
                )
                if variant
            )
        )
        words = [word for variant in variants for word in variant.split()]
        if not words:
            return False

        for alias in aliases:
            alias_normalized = self._normalize(alias)
            alias_variants = tuple(
                dict.fromkeys(
                    variant
                    for variant in (
                        alias_normalized,
                        self._squash_repeats(alias_normalized, 2),
                        self._squash_repeats(alias_normalized, 1),
                    )
                    if variant
                )
            )

            if " " in alias_normalized:
                if any(alias_variant in variant for variant in variants for alias_variant in alias_variants):
                    return True
                window_size = len(alias_normalized.split())
                if any(self._unsafe_alias_score(window, alias_normalized) >= threshold for window in self._token_windows(words, window_size)):
                    return True
                continue

            if alias_normalized in words:
                return True
            if len(alias_normalized) <= 3:
                continue
            if any(len(word) > 3 and self._unsafe_alias_score(word, alias_normalized) >= threshold for word in words):
                return True
        return False

    def _unsafe_alias_score(self, text: str, alias: str) -> int:
        text_variants = (
            text,
            self._squash_repeats(text, 2),
            self._squash_repeats(text, 1),
        )
        alias_variants = (
            alias,
            self._squash_repeats(alias, 2),
            self._squash_repeats(alias, 1),
        )
        best = 0
        for text_variant in dict.fromkeys(variant for variant in text_variants if variant):
            for alias_variant in dict.fromkeys(variant for variant in alias_variants if variant):
                if text_variant == alias_variant:
                    best = max(best, 100)
                elif fuzz:
                    best = max(best, int(max(fuzz.ratio(text_variant, alias_variant), fuzz.WRatio(text_variant, alias_variant), fuzz.token_set_ratio(text_variant, alias_variant))))
                else:
                    best = max(best, int(SequenceMatcher(None, text_variant, alias_variant).ratio() * 100))
        return min(best, 100)

    def _score(self, text: str, alias: str) -> int:
        if fuzz:
            return int(max(fuzz.WRatio(text, alias), fuzz.partial_ratio(text, alias), fuzz.token_set_ratio(text, alias)))
        return int(SequenceMatcher(None, text, alias).ratio() * 100)

    def _alias_score(self, text: str, alias: str) -> int:
        text_variants = self._match_variants(text)
        alias_variants = self._match_variants(alias)
        if not text_variants or not alias_variants:
            return 0

        best = 0
        for text_variant in text_variants:
            text_compact = self._compact(text_variant)
            text_signature = self._consonant_signature(text_variant)
            for alias_variant in alias_variants:
                alias_compact = self._compact(alias_variant)
                alias_signature = self._consonant_signature(alias_variant)
                best = max(best, self._score(text_variant, alias_variant))
                if text_compact and text_compact == alias_compact:
                    best = max(best, 100)
                if text_compact and alias_compact:
                    best = max(best, self._score(text_compact, alias_compact))
                if len(text_signature) >= 3 and text_signature == alias_signature:
                    best = max(best, 96)
                elif len(text_signature) >= 3 and len(alias_signature) >= 3:
                    signature_score = self._score(text_signature, alias_signature)
                    if signature_score >= 90 and best >= 62:
                        best = max(best, 90)
        return min(best, 100)

    def _match_variants(self, text: str) -> tuple[str, ...]:
        normalized = self._normalize(text)
        variants = [
            normalized,
            self._squash_repeats(normalized, 2),
            self._squash_repeats(normalized, 1),
            self._compact(normalized),
            self._compact(self._squash_repeats(normalized, 2)),
            self._compact(self._squash_repeats(normalized, 1)),
        ]
        return tuple(dict.fromkeys(variant for variant in variants if variant))

    def _squash_repeats(self, text: str, limit: int) -> str:
        if limit < 1:
            return text
        return re.sub(r"(.)\1+", lambda match: match.group(1) * min(len(match.group(0)), limit), text)

    def _compact(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    def _consonant_signature(self, text: str) -> str:
        compact = self._compact(self._squash_repeats(text, 1))
        return re.sub(r"[aeiou]+", "", compact)

    def _token_windows(self, words: list[str], size: int) -> list[str]:
        if size <= 0 or not words:
            return []
        if len(words) < size:
            return [" ".join(words)]
        return [" ".join(words[index : index + size]) for index in range(0, len(words) - size + 1)]

    def _resolution_meta(self, source: str, confidence: float, matched_alias: str, normalized: str) -> dict[str, Any]:
        return {
            "source": source,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "matched_alias": matched_alias,
            "normalized_prompt": normalized,
        }

    def _legacy_request_from_text(self, text: str, confirmed: bool = False) -> AgentCommandRequest | None:
        normalized = self._normalize(text)
        if not normalized:
            return None

        if re.search(r"\b(delete|remove|wipe|erase|rm|rmdir|format)\b", normalized) and not re.search(r"\bmemory\b", normalized):
            return AgentCommandRequest(command_id="blocked_action", input_text=text, params={"reason": "destructive file/system request"}, confirmed=confirmed)
        if re.search(r"\b(click|type|move|close|control|press|shell|terminal|cmd|powershell|secret|api key)\b", normalized):
            return AgentCommandRequest(command_id="blocked_action", input_text=text, params={"reason": "free-form desktop or shell control"}, confirmed=confirmed)

        if "frontend" in normalized and "lint" in normalized:
            return AgentCommandRequest(command_id="run_frontend_lint", input_text=text, confirmed=confirmed)
        if ("backend" in normalized or "pytest" in normalized) and re.search(r"\b(test|tests|pytest)\b", normalized):
            return AgentCommandRequest(command_id="run_backend_tests", input_text=text, confirmed=confirmed)
        if "frontend" in normalized and re.search(r"\b(start|run|launch)\b", normalized) and re.search(r"\b(server|dev)\b", normalized):
            return AgentCommandRequest(command_id="start_frontend_dev_server", input_text=text, confirmed=confirmed)

        if "health" in normalized:
            return AgentCommandRequest(command_id="check_backend_health", input_text=text, confirmed=confirmed)
        if "provider" in normalized or "setup" in normalized:
            return AgentCommandRequest(command_id="check_provider_setup", input_text=text, confirmed=confirmed)
        if "latest report" in normalized and re.search(r"\b(open|show|view)\b", normalized):
            return AgentCommandRequest(command_id="open_latest_report", input_text=text, confirmed=confirmed)
        if "report" in normalized and re.search(r"\b(list|show)\b", normalized):
            return AgentCommandRequest(command_id="list_reports", input_text=text, confirmed=confirmed)
        if "document" in normalized and re.search(r"\b(list|show)\b", normalized):
            return AgentCommandRequest(command_id="list_documents", input_text=text, confirmed=confirmed)
        if ("study" in normalized or "artifact" in normalized) and re.search(r"\b(list|show)\b", normalized):
            return AgentCommandRequest(command_id="list_study_artifacts", input_text=text, confirmed=confirmed)
        if "memory" in normalized and re.search(r"\b(list|show)\b", normalized):
            return AgentCommandRequest(command_id="list_memory", input_text=text, confirmed=confirmed)

        remember_match = re.match(r"^(please\s+)?remember\s+(?P<text>.+)$", normalized)
        if remember_match:
            memory_text = re.sub(r"^(please\s+)?remember\s+", "", text.strip(), flags=re.IGNORECASE).strip()
            return AgentCommandRequest(
                command_id="save_memory",
                input_text=text,
                params={"category": self._infer_memory_category(memory_text), "text": memory_text},
                confirmed=confirmed,
            )

        study_type = self._infer_study_type(normalized)
        if study_type:
            topic = re.sub(
                r"^\s*(please\s+)?(generate|create|make|build)\s+(me\s+)?(study\s+)?(notes|flashcards|quiz|revision plan|viva questions)\s*(for|on|about)?\s*",
                "",
                text.strip(),
                flags=re.IGNORECASE,
            ).strip(" .")
            return AgentCommandRequest(
                command_id="generate_study_artifact",
                input_text=text,
                params={"artifact_type": study_type, "topic": topic or text.strip()},
                confirmed=confirmed,
            )

        if re.search(r"\b(open|launch|start|go to|visit)\b", normalized):
            direct_url = self._extract_direct_url(text)
            if direct_url:
                return AgentCommandRequest(command_id="open_arbitrary_url", input_text=text, params={"url": direct_url}, confirmed=confirmed)

            target_key = self._detect_allowed_target(normalized)
            if target_key:
                return AgentCommandRequest(command_id="open_allowlisted_target", input_text=text, params={"target": target_key}, confirmed=confirmed)

        return None

    async def execute(self, request: AgentCommandRequest) -> AgentCommandResponse:
        resolution = dict(request.resolution or {})
        command = self.commands.get(request.command_id)
        if not command:
            return self._response(
                command_id=request.command_id,
                label="Unknown Command",
                risk="blocked",
                outcome="blocked",
                message="That command is not in Astra's safe registry.",
                safety_decision="unknown_command_blocked",
                input_text=request.input_text,
                params=request.params,
                resolution=resolution,
            )

        params, validation_error = self._validate_params(command, request.params)
        if validation_error:
            return self._response(
                command_id=command.id,
                label=command.label,
                risk=command.risk,
                outcome="blocked",
                message=validation_error,
                safety_decision="invalid_params_blocked",
                input_text=request.input_text,
                params=request.params,
                resolution=resolution,
            )

        if command.risk == "blocked":
            return self._response(
                command_id=command.id,
                label=command.label,
                risk=command.risk,
                outcome="blocked",
                message=f"Blocked: {params.get('reason') or command.description}",
                safety_decision="blocked_policy",
                input_text=request.input_text,
                params=params,
                resolution=resolution,
            )

        passed, test_message = await self._run_self_test(command)
        if not passed:
            return self._response(
                command_id=command.id,
                label=command.label,
                risk=command.risk,
                outcome="failure",
                message=f"Self-test failed before execution: {test_message}",
                safety_decision="self_test_failed",
                input_text=request.input_text,
                params=params,
                resolution=resolution,
            )

        if resolution.get("needs_confirmation") and not request.confirmed:
            return self._response(
                command_id=command.id,
                label=command.label,
                risk=command.risk,
                outcome="confirmation_required",
                message=self._intent_confirmation_message(command, params, resolution),
                safety_decision="intent_confirmation_required",
                input_text=request.input_text,
                params=params,
                confirmation_required=True,
                resolution=resolution,
            )

        if command.risk == "safe_confirm" and not request.confirmed:
            return self._response(
                command_id=command.id,
                label=command.label,
                risk=command.risk,
                outcome="confirmation_required",
                message=self._safe_confirm_message(command, params, resolution),
                safety_decision="confirmation_required",
                input_text=request.input_text,
                params=params,
                confirmation_required=True,
                resolution=resolution,
            )

        if not command.executor:
            return self._response(
                command_id=command.id,
                label=command.label,
                risk=command.risk,
                outcome="planned",
                message="Astra prepared a plan, but this registry entry has no executor.",
                safety_decision="plan_only",
                input_text=request.input_text,
                params=params,
                resolution=resolution,
            )

        ok, message, data = await command.executor(params)
        if ok:
            saved_correction = self._maybe_save_intent_correction(command, params, resolution, request.input_text, request.confirmed)
            if saved_correction:
                resolution["correction_saved"] = True
            message = self._resolved_success_message(command, params, message, resolution)
        if resolution:
            data = {**data, "resolution": resolution}
        return self._response(
            command_id=command.id,
            label=command.label,
            risk=command.risk,
            outcome="success" if ok else "failure",
            message=message,
            safety_decision="executed_after_validation" if ok else "executor_failed",
            input_text=request.input_text,
            params=params,
            data=data,
            resolution=resolution,
        )

    def _intent_confirmation_message(self, command: AgentCommandDefinition, params: dict[str, Any], resolution: dict[str, Any]) -> str:
        label = self._resolved_label(command, params)
        confidence = resolution.get("confidence")
        suffix = f" ({round(float(confidence) * 100)}% match)" if isinstance(confidence, (int, float)) else ""
        return f"Did you mean {label}? Confirm to run it{suffix}."

    def _safe_confirm_message(self, command: AgentCommandDefinition, params: dict[str, Any], resolution: dict[str, Any]) -> str:
        if resolution:
            return f"I understood this as {self._resolved_label(command, params)}. Confirmation required before Astra runs it."
        return f"Confirmation required before Astra runs: {command.label}."

    def _resolved_success_message(self, command: AgentCommandDefinition, params: dict[str, Any], message: str, resolution: dict[str, Any]) -> str:
        source = resolution.get("source")
        if source in {"fuzzy", "llm", "correction"}:
            return f"I understood \"{resolution.get('normalized_prompt', '')}\" as {self._resolved_label(command, params)}. {message}"
        if source in {"semantic_llm", "semantic_local"} and command.id == "generate_mock_test":
            return f"I understood this as {self._resolved_label(command, params)}. {message}"
        return message

    def _resolved_label(self, command: AgentCommandDefinition, params: dict[str, Any]) -> str:
        if command.id == "open_allowlisted_target":
            if params.get("target") == "file_explorer" and params.get("drive"):
                return f"File Explorer ({str(params['drive']).upper()}:)"
            target = self.safe_targets.get(str(params.get("target")))
            return target.label if target else command.label
        if command.id == "generate_study_artifact":
            artifact = str(params.get("artifact_type", "study artifact")).replace("_", " ")
            return f"Generate {artifact.title()}"
        if command.id == "generate_mock_test":
            topic = str(params.get("topic") or "topic").strip()
            return f"Generate Mock Test for {topic}"
        if command.id == "open_latest_mock_test":
            topic = str(params.get("topic") or "").strip()
            return f"Open Latest Mock Test for {topic}" if topic else "Open Latest Mock Test"
        return command.label

    def _build_commands(self) -> dict[str, AgentCommandDefinition]:
        commands = [
            AgentCommandDefinition(
                id="check_backend_health",
                label="Check Backend Health",
                description="Check API model and provider availability.",
                category="System",
                risk="safe_auto",
                executor=self._exec_backend_health,
                tester=self._test_backend_ready,
            ),
            AgentCommandDefinition(
                id="check_provider_setup",
                label="Check Provider Setup",
                description="Show configured and missing AI/search/voice providers.",
                category="System",
                risk="safe_auto",
                executor=self._exec_provider_setup,
                tester=self._test_backend_ready,
            ),
            AgentCommandDefinition(
                id="list_reports",
                label="List Reports",
                description="List saved Markdown research reports.",
                category="Library",
                risk="safe_auto",
                executor=self._exec_list_reports,
                tester=self._test_reports_ready,
            ),
            AgentCommandDefinition(
                id="open_latest_report",
                label="Open Latest Report",
                description="Return the latest research report link.",
                category="Library",
                risk="safe_auto",
                executor=self._exec_open_latest_report,
                tester=self._test_reports_ready,
            ),
            AgentCommandDefinition(
                id="list_documents",
                label="List Documents",
                description="List uploaded PDF documents.",
                category="Documents",
                risk="safe_auto",
                executor=self._exec_list_documents,
                tester=self._test_documents_ready,
            ),
            AgentCommandDefinition(
                id="list_study_artifacts",
                label="List Study Artifacts",
                description="List generated notes, flashcards, quizzes, and revision plans.",
                category="Study",
                risk="safe_auto",
                executor=self._exec_list_study_artifacts,
                tester=self._test_study_ready,
            ),
            AgentCommandDefinition(
                id="list_mock_tests",
                label="List Mock Tests",
                description="List generated mock tests.",
                category="Study",
                risk="safe_auto",
                executor=self._exec_list_mock_tests,
                tester=self._test_mock_tests_ready,
            ),
            AgentCommandDefinition(
                id="open_latest_mock_test",
                label="Open Latest Mock Test",
                description="Return the latest generated mock test so the UI can open it.",
                category="Study",
                risk="safe_auto",
                params_schema={"topic": {"type": "string"}},
                executor=self._exec_open_latest_mock_test,
                tester=self._test_mock_tests_ready,
            ),
            AgentCommandDefinition(
                id="list_memory",
                label="List Memory",
                description="List explicit memories saved by the user.",
                category="Memory",
                risk="safe_auto",
                executor=self._exec_list_memory,
                tester=self._test_memory_ready,
            ),
            AgentCommandDefinition(
                id="open_allowlisted_target",
                label="Open Approved App or Site",
                description="Open one approved site or desktop app.",
                category="Desktop",
                risk="safe_auto",
                params_schema={"target": {"type": "string", "enum": sorted(self.safe_targets)}, "drive": {"type": "string", "pattern": "^[A-Z]$"}},
                required_params=("target",),
                executor=self._exec_open_allowlisted_target,
                tester=self._test_desktop_ready,
            ),
            AgentCommandDefinition(
                id="run_frontend_lint",
                label="Run Frontend Lint",
                description="Run npm lint in astra-frontend.",
                category="Project",
                risk="safe_confirm",
                executor=self._exec_frontend_lint,
                tester=self._test_frontend_ready,
            ),
            AgentCommandDefinition(
                id="run_backend_tests",
                label="Run Backend Tests",
                description="Run pytest in astra-backend.",
                category="Project",
                risk="safe_confirm",
                executor=self._exec_backend_tests,
                tester=self._test_pytest_ready,
            ),
            AgentCommandDefinition(
                id="start_frontend_dev_server",
                label="Start Frontend Dev Server",
                description="Start the Next.js dev server if it is not already running.",
                category="Project",
                risk="safe_confirm",
                executor=self._exec_start_frontend,
                tester=self._test_frontend_ready,
            ),
            AgentCommandDefinition(
                id="open_arbitrary_url",
                label="Open URL",
                description="Open a user-provided http(s) URL after confirmation.",
                category="Desktop",
                risk="safe_confirm",
                params_schema={"url": {"type": "string", "format": "uri"}},
                required_params=("url",),
                executor=self._exec_open_arbitrary_url,
                tester=self._test_desktop_ready,
            ),
            AgentCommandDefinition(
                id="save_memory",
                label="Save Memory",
                description="Save an explicit user memory.",
                category="Memory",
                risk="safe_confirm",
                params_schema={"category": {"type": "string"}, "text": {"type": "string"}},
                required_params=("text",),
                executor=self._exec_save_memory,
                tester=self._test_memory_ready,
            ),
            AgentCommandDefinition(
                id="delete_memory",
                label="Delete Memory",
                description="Delete an explicit memory by id.",
                category="Memory",
                risk="safe_confirm",
                params_schema={"memory_id": {"type": "string"}},
                required_params=("memory_id",),
                executor=self._exec_delete_memory,
                tester=self._test_memory_ready,
            ),
            AgentCommandDefinition(
                id="generate_study_artifact",
                label="Generate Study Artifact",
                description="Create notes, flashcards, quiz, revision plan, or viva questions.",
                category="Study",
                risk="safe_confirm",
                params_schema={"artifact_type": {"type": "string"}, "topic": {"type": "string"}},
                required_params=("artifact_type", "topic"),
                executor=self._exec_generate_study_artifact,
                tester=self._test_study_ready,
            ),
            AgentCommandDefinition(
                id="generate_mock_test",
                label="Generate Mock Test",
                description="Create a local MCQ mock test from a topic or source-backed PYQ material.",
                category="Study",
                risk="safe_confirm",
                params_schema={
                    "topic": {"type": "string"},
                    "exam": {"type": "string"},
                    "subject": {"type": "string"},
                    "question_count": {"type": "integer", "default": 10},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "mixed"]},
                    "mode": {"type": "string", "enum": ["mcq"]},
                    "duration_minutes": {"type": "integer", "default": 20},
                    "source_requirement": {"type": "string", "enum": ["none", "pyq_required", "source_backed"]},
                    "source_mode": {"type": "string", "enum": ["uploaded_docs"]},
                    "source_query": {"type": "string"},
                    "constraints": {"type": "array"},
                },
                required_params=("topic",),
                executor=self._exec_generate_mock_test,
                tester=self._test_mock_tests_ready,
            ),
            AgentCommandDefinition(
                id="clarify_agent_intent",
                label="Clarify Agent Intent",
                description="Ask for a safe clarification before running a source-sensitive action.",
                category="Safety",
                risk="safe_auto",
                params_schema={"message": {"type": "string"}, "actions": {"type": "array"}},
                required_params=("message",),
                executor=self._exec_clarify_agent_intent,
                tester=self._test_backend_ready,
            ),
            AgentCommandDefinition(
                id="blocked_action",
                label="Blocked Action",
                description="Reject unsafe free-form desktop, shell, secret, or destructive requests.",
                category="Safety",
                risk="blocked",
                params_schema={"reason": {"type": "string"}},
                executor=None,
                tester=self._test_backend_ready,
            ),
        ]
        return {command.id: command for command in commands}

    async def _exec_backend_health(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        providers = self._provider_status()
        return True, "Backend is online and provider status is ready.", {
            "model": self.settings.resolved_cerebras_pro_model,
            "models": self.settings.cerebras_models,
            "providers": providers,
        }

    async def _exec_provider_setup(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        providers = self._provider_status()
        missing = [name for name, enabled in providers.items() if not enabled and name in {"cerebras", "tavily", "piper_local"}]
        message = "All core providers are configured." if not missing else f"Missing setup: {', '.join(missing)}."
        return True, message, {"providers": providers, "missing": missing}

    async def _exec_list_reports(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        reports = [self._report_summary(report) for report in self.reports.list_reports()]
        return True, f"Found {len(reports)} saved reports.", {"reports": reports}

    async def _exec_open_latest_report(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        reports = self.reports.list_reports()
        if not reports:
            return False, "No saved reports are available yet.", {"reports": []}
        latest = self._report_summary(reports[0])
        return True, f"Latest report is ready: {latest['title']}.", {"report": latest}

    async def _exec_list_documents(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        documents = [item.model_dump(mode="json") for item in self.documents.list_documents()]
        return True, f"Found {len(documents)} uploaded documents.", {"documents": documents}

    async def _exec_list_study_artifacts(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        artifacts = [self._artifact_summary(item) for item in self.study.list_artifacts()]
        return True, f"Found {len(artifacts)} study artifacts.", {"artifacts": artifacts}

    async def _exec_list_mock_tests(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        tests = [item.model_dump(mode="json") for item in self.mock_tests.list_tests()]
        return True, f"Found {len(tests)} mock tests.", {"mock_tests": tests, "mock_test": tests[0] if tests else None}

    async def _exec_open_latest_mock_test(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        tests = self.mock_tests.list_tests()
        topic = str(params.get("topic") or "").strip()
        selected = self._select_mock_test(tests, topic)
        data = {"mock_tests": [item.model_dump(mode="json") for item in tests[:10]], "open_panel": "mock_test"}
        if not selected:
            suffix = f" for {topic}" if topic else ""
            return True, f"I could not find a saved mock test{suffix} yet.", {**data, "mock_test": None}
        selected_data = selected.model_dump(mode="json")
        selected_topic = "" if self._mock_topic_looks_like_status(str(selected.topic)) else selected.topic
        topic_label = topic or selected_topic
        label = f" related to {topic_label}" if topic_label else ""
        return True, f"I found your latest mock test{label}. Opening Mock Test.", {**data, "mock_test": selected_data}

    async def _exec_list_memory(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        items = [item.model_dump(mode="json") for item in self.memory.list_items()]
        return True, f"Found {len(items)} saved memories.", {"memory": items}

    async def _exec_open_allowlisted_target(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        target = self.safe_targets[params["target"]]
        if params["target"] == "file_explorer":
            drive = str(params.get("drive") or "").strip() or None
            result = await self.desktop.open_file_explorer(drive)
            return result.ok, result.message, {"target": result.target, "action": result.action, "drive": drive}
        result = await self.desktop.open_target(target)
        return result.ok, result.message, {"target": result.target, "action": result.action}

    async def _exec_frontend_lint(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        return await self._run_fixed_command(["npm", "run", "lint"], self.frontend_dir, "Frontend lint")

    async def _exec_backend_tests(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        return await self._run_fixed_command([sys.executable, "-m", "pytest"], self.backend_dir, "Backend tests")

    async def _exec_start_frontend(self, _: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        if await self._url_responds("http://127.0.0.1:3000"):
            return True, "Frontend dev server is already responding on http://localhost:3000.", {"url": "http://localhost:3000"}

        stdout = (self.frontend_dir / "dev-server.log").open("a", encoding="utf-8")
        stderr = (self.frontend_dir / "dev-server.err.log").open("a", encoding="utf-8")
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
        try:
            subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=self.frontend_dir,
                stdout=stdout,
                stderr=stderr,
                creationflags=creation_flags,
            )
        except Exception as exc:
            stdout.close()
            stderr.close()
            return False, f"Could not start frontend dev server: {exc}", {}
        return True, "Started the frontend dev server. Open http://localhost:3000.", {"url": "http://localhost:3000"}

    async def _exec_open_arbitrary_url(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        url = params["url"]
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:
            return False, f"Windows could not open that URL: {exc}", {"url": url}
        return True, f"Opened {url}.", {"url": url}

    async def _exec_save_memory(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        item = self.memory.create(AgentMemoryCreateRequest(category=params.get("category", "general"), text=params["text"]))
        return True, "Saved that memory.", {"memory": item.model_dump(mode="json")}

    async def _exec_delete_memory(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        deleted = self.memory.delete(params["memory_id"])
        if not deleted:
            return False, "Memory item was not found.", {"memory_id": params["memory_id"]}
        return True, "Deleted that memory.", {"memory_id": params["memory_id"]}

    async def _exec_generate_study_artifact(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        request = StudyGenerateRequest(
            artifact_type=params["artifact_type"],
            topic=params["topic"],
            source_text=params.get("source_text", ""),
            report_id=params.get("report_id"),
            document_id=params.get("document_id"),
        )
        artifact, setup = await self.study.generate(request)
        return True, f"Generated {artifact.artifact_type.replace('_', ' ')} for {request.topic}.", {"artifact": artifact.model_dump(mode="json"), "setup_required": setup}

    async def _exec_generate_mock_test(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        request = MockTestGenerateRequest(
            topic=params["topic"],
            exam=params.get("exam", ""),
            subject=params.get("subject", ""),
            question_count=int(params.get("question_count", 10)),
            difficulty=params.get("difficulty", "mixed"),
            mode="mcq",
            duration_minutes=int(params.get("duration_minutes", 20)),
            source_requirement=params.get("source_requirement", "none"),
            source_mode="uploaded_docs",
            source_query=params.get("source_query", ""),
            constraints=params.get("constraints", []),
        )
        try:
            test, setup = await self.mock_tests.generate(request)
        except MockTestSourceMaterialError as exc:
            return False, str(exc), {"setup_required": [], "source_actions": exc.source_actions}
        except ValueError as exc:
            return False, str(exc), {"setup_required": []}
        return (
            True,
            f"Generated a {test.question_count}-question mock test for {test.topic}.",
            {"mock_test": test.model_dump(mode="json"), "setup_required": setup},
        )

    async def _exec_clarify_agent_intent(self, params: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        actions = params.get("actions") if isinstance(params.get("actions"), list) else []
        return True, str(params["message"]), {"actions": actions, "topic": params.get("topic", "")}

    async def _run_fixed_command(self, argv: list[str], cwd: Path, label: str) -> tuple[bool, str, dict[str, Any]]:
        if not cwd.exists():
            return False, f"{label} cannot run because {cwd} does not exist.", {"cwd": str(cwd)}

        def run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=120,
            )

        try:
            completed = await asyncio.to_thread(run)
        except subprocess.TimeoutExpired:
            return False, f"{label} timed out.", {"argv": argv, "cwd": str(cwd)}
        except Exception as exc:
            return False, f"{label} could not run: {exc}", {"argv": argv, "cwd": str(cwd)}

        output = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        ok = completed.returncode == 0
        return ok, f"{label} {'passed' if ok else 'failed'} with exit code {completed.returncode}.", {
            "argv": argv,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "output": output[-5000:],
        }

    async def _test_backend_ready(self) -> tuple[bool, str]:
        return True, "Backend service is available."

    async def _test_reports_ready(self) -> tuple[bool, str]:
        return (self.reports.reports_dir.exists(), "Reports directory is available.")

    async def _test_documents_ready(self) -> tuple[bool, str]:
        return (self.documents.documents_dir.exists(), "Documents directory is available.")

    async def _test_study_ready(self) -> tuple[bool, str]:
        return (self.study.study_dir.exists(), "Study artifact directory is available.")

    async def _test_mock_tests_ready(self) -> tuple[bool, str]:
        return (self.mock_tests.mock_tests_dir.exists(), "Mock test directory is available.")

    async def _test_memory_ready(self) -> tuple[bool, str]:
        return (self.memory.memory_dir.exists(), "Memory store is available.")

    async def _test_desktop_ready(self) -> tuple[bool, str]:
        return (bool(self.safe_targets), "Approved desktop targets are registered.")

    async def _test_frontend_ready(self) -> tuple[bool, str]:
        package_json = self.frontend_dir / "package.json"
        return (package_json.exists(), "Frontend package is available." if package_json.exists() else "Frontend package.json was not found.")

    async def _test_pytest_ready(self) -> tuple[bool, str]:
        try:
            import pytest  # noqa: F401
        except Exception:
            return False, "pytest is not installed in the backend environment."
        return True, "pytest is available."

    async def _run_self_test(self, command: AgentCommandDefinition) -> tuple[bool, str]:
        tester = command.tester or self._test_backend_ready
        try:
            passed, message = await tester()
        except Exception as exc:
            passed, message = False, str(exc)

        results = self._read_test_results()
        results[command.id] = {
            "test_status": "passed" if passed else "failed",
            "last_tested_at": datetime.utcnow().isoformat(),
            "test_message": message,
        }
        self._write_test_results(results)
        return passed, message

    def _validate_params(self, command: AgentCommandDefinition, params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        clean = dict(params or {})
        for key in command.required_params:
            if key not in clean or clean[key] in {None, ""}:
                return clean, f"Missing required parameter: {key}."

        if command.id == "open_allowlisted_target":
            if clean.get("target") not in self.safe_targets:
                return clean, "That target is not approved for automatic opening."
            if "drive" in clean and clean.get("drive") not in {None, ""}:
                if clean.get("target") != "file_explorer":
                    return clean, "Local drive selection is only allowed with File Explorer."
                drive = str(clean.get("drive", "")).strip().rstrip(":\\/").upper()
                if not re.fullmatch(r"[A-Z]", drive):
                    return clean, "Only a local drive root like E: can be opened."
                clean["drive"] = drive

        if command.id == "open_arbitrary_url":
            url = str(clean.get("url", "")).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return clean, "Only valid http(s) URLs can be opened."
            clean["url"] = url

        if command.id == "open_latest_mock_test":
            topic = str(clean.get("topic", "")).strip(" .")
            if topic:
                clean["topic"] = topic
            elif "topic" in clean:
                clean.pop("topic", None)

        if command.id == "save_memory":
            text = str(clean.get("text", "")).strip()
            if not text:
                return clean, "Memory text cannot be empty."
            category = str(clean.get("category", "general")).strip() or "general"
            if category not in {"course", "project", "goal", "preference", "general"}:
                category = "general"
            clean["text"] = text
            clean["category"] = category

        if command.id == "generate_study_artifact":
            artifact_type = str(clean.get("artifact_type", "")).strip()
            if artifact_type not in {"notes", "flashcards", "quiz", "revision_plan", "viva_questions"}:
                return clean, "Unsupported study artifact type."
            clean["topic"] = str(clean.get("topic", "")).strip()
            if not clean["topic"]:
                return clean, "Study artifact topic cannot be empty."

        if command.id == "generate_mock_test":
            clean["topic"] = str(clean.get("topic", "")).strip(" .")
            if not clean["topic"]:
                return clean, "Mock test topic cannot be empty."
            if self._mock_topic_looks_like_status(clean["topic"]):
                return clean, "That looks like a mock-test status request, not a new test topic."
            clean["exam"] = str(clean.get("exam") or "").strip(" .")[:80]
            clean["subject"] = str(clean.get("subject") or "").strip(" .")[:80]
            try:
                clean["question_count"] = max(1, min(50, int(clean.get("question_count", 10))))
            except (TypeError, ValueError):
                clean["question_count"] = 10
            difficulty = str(clean.get("difficulty", "mixed")).strip().lower()
            clean["difficulty"] = difficulty if difficulty in {"easy", "medium", "hard", "mixed"} else "mixed"
            clean["mode"] = "mcq"
            try:
                clean["duration_minutes"] = max(1, min(180, int(clean.get("duration_minutes", 20))))
            except (TypeError, ValueError):
                clean["duration_minutes"] = 20
            source_requirement = str(clean.get("source_requirement", "none")).strip().lower()
            clean["source_requirement"] = source_requirement if source_requirement in {"none", "pyq_required", "source_backed"} else "none"
            clean["source_mode"] = "uploaded_docs"
            clean["source_query"] = str(clean.get("source_query") or "").strip()[:240]
            constraints = clean.get("constraints")
            clean["constraints"] = [str(item).strip()[:80] for item in constraints if str(item).strip()] if isinstance(constraints, list) else []

        if command.id == "clarify_agent_intent":
            message = str(clean.get("message", "")).strip()
            if not message:
                return clean, "Clarification message cannot be empty."
            clean["message"] = message
            actions = clean.get("actions")
            clean["actions"] = [str(item).strip()[:80] for item in actions if str(item).strip()] if isinstance(actions, list) else []
            clean["topic"] = str(clean.get("topic", "")).strip(" .")

        return clean, None

    def _response(
        self,
        command_id: str,
        label: str,
        risk: AgentCommandRisk,
        outcome: AgentCommandOutcome,
        message: str,
        safety_decision: str,
        input_text: str,
        params: dict[str, Any],
        data: dict[str, Any] | None = None,
        confirmation_required: bool = False,
        resolution: dict[str, Any] | None = None,
    ) -> AgentCommandResponse:
        resolution_data = dict(resolution or {})
        audit = AgentAuditEntry(
            id=uuid.uuid4().hex,
            command_id=command_id,
            label=label,
            input_text=input_text,
            params=params,
            outcome=outcome,
            safety_decision=safety_decision,
            message=message,
            resolution=resolution_data,
        )
        self._append_audit(audit)
        status = "complete" if outcome == "success" else "warning" if outcome in {"blocked", "confirmation_required", "planned"} else "error"
        return AgentCommandResponse(
            command_id=command_id,
            label=label,
            risk=risk,
            outcome=outcome,
            message=message,
            confirmation_required=confirmation_required,
            params=params,
            data=data or {},
            events=[AgentEvent(agent="Safety Validator", status=status, message=message)],
            audit=audit,
            resolution=resolution_data,
        )

    def _append_audit(self, entry: AgentAuditEntry) -> None:
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")

    def _read_intent_corrections(self) -> dict[str, Any]:
        if not self.correction_path.exists():
            return {}
        try:
            raw = json.loads(self.correction_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_intent_corrections(self, corrections: dict[str, Any]) -> None:
        self.correction_path.write_text(json.dumps(corrections, indent=2, ensure_ascii=True), encoding="utf-8")

    def _mark_intent_correction_used(self, normalized: str, correction: dict[str, Any]) -> None:
        corrections = self._read_intent_corrections()
        current = corrections.get(normalized)
        if not isinstance(current, dict):
            current = correction
        current["uses"] = int(current.get("uses") or 0) + 1
        current["last_used_at"] = datetime.utcnow().isoformat()
        corrections[normalized] = current
        self._write_intent_corrections(corrections)

    def _maybe_save_intent_correction(
        self,
        command: AgentCommandDefinition,
        params: dict[str, Any],
        resolution: dict[str, Any],
        input_text: str,
        confirmed: bool,
    ) -> bool:
        source = resolution.get("source")
        confidence = resolution.get("confidence")
        normalized = str(resolution.get("normalized_prompt") or self._normalize(input_text))
        if not confirmed or source not in {"fuzzy", "llm"} or not normalized:
            return False
        try:
            numeric_confidence = float(confidence)
        except (TypeError, ValueError):
            numeric_confidence = 0.0
        if not resolution.get("needs_confirmation") and numeric_confidence >= 0.92:
            return False

        clean_params, validation_error = self._validate_params(command, dict(params))
        if validation_error:
            return False

        corrections = self._read_intent_corrections()
        previous = corrections.get(normalized) if isinstance(corrections.get(normalized), dict) else {}
        now = datetime.utcnow().isoformat()
        corrections[normalized] = {
            "command_id": command.id,
            "params": clean_params,
            "matched_alias": str(resolution.get("matched_alias") or self._resolved_label(command, params)),
            "confidence": round(max(0.0, min(1.0, numeric_confidence)), 3),
            "created_at": previous.get("created_at") or now,
            "updated_at": now,
            "uses": int(previous.get("uses") or 0),
        }
        self._write_intent_corrections(corrections)
        return True

    def _ability_for(self, command: AgentCommandDefinition, test_result: dict[str, Any]) -> AgentAbility:
        return AgentAbility(
            id=command.id,
            label=command.label,
            description=command.description,
            category=command.category,
            risk=command.risk,
            params_schema=command.params_schema,
            test_status=test_result.get("test_status", "untested"),
            last_tested_at=test_result.get("last_tested_at"),
            test_message=test_result.get("test_message", ""),
        )

    def _read_test_results(self) -> dict[str, dict[str, Any]]:
        if not self.test_path.exists():
            return {}
        try:
            raw = json.loads(self.test_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_test_results(self, results: dict[str, dict[str, Any]]) -> None:
        self.test_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    def _build_intent_candidates(self) -> list[IntentCandidate]:
        candidates: list[IntentCandidate] = []

        def add(command_id: str, label: str, aliases: tuple[str, ...], params: dict[str, Any] | None = None) -> None:
            candidates.append(IntentCandidate(command_id=command_id, label=label, aliases=aliases, params=params or {}))

        for key, target in self.safe_targets.items():
            aliases = {target.label, *target.aliases}
            for alias in list(aliases):
                aliases.update({f"open {alias}", f"launch {alias}", f"start {alias}", f"go to {alias}", f"visit {alias}"})
            if key == "file_explorer":
                aliases.update(
                    {
                        "file manager",
                        "open file manager",
                        "windows explorer",
                        "open windows explorer",
                        "folders",
                        "open folder",
                        "open my files",
                    }
                )
            add("open_allowlisted_target", f"Open {target.label}", tuple(sorted(aliases)), {"target": key})

        add("check_backend_health", "Check Backend Health", ("health", "backend health", "check health", "server health", "api health"))
        add("check_provider_setup", "Check Provider Setup", ("provider setup", "api setup", "check providers", "check setup", "provider status"))
        add("list_reports", "List Reports", ("list reports", "show reports", "saved reports", "research reports"))
        add("open_latest_report", "Open Latest Report", ("open latest report", "show latest report", "view latest report", "latest report"))
        add("list_documents", "List Documents", ("list documents", "show documents", "uploaded documents", "pdf documents"))
        add("list_study_artifacts", "List Study Artifacts", ("list study artifacts", "show study artifacts", "study artifacts", "study materials"))
        add("list_mock_tests", "List Mock Tests", ("list mock tests", "show mock tests", "all mock tests", "saved mock tests", "mock tests"))
        add(
            "open_latest_mock_test",
            "Open Latest Mock Test",
            (
                "open mock test",
                "show mock test",
                "view mock test",
                "where is my mock test",
                "latest mock test",
                "mock test not created",
                "mock not created",
                "find mock test",
            ),
        )
        add("list_memory", "List Memory", ("list memory", "show memory", "saved memory", "memories"))
        add("run_frontend_lint", "Run Frontend Lint", ("run frontend lint", "frontend lint", "npm lint", "lint frontend"))
        add("run_backend_tests", "Run Backend Tests", ("run backend tests", "backend tests", "pytest", "run pytest"))
        add("start_frontend_dev_server", "Start Frontend Dev Server", ("start frontend dev server", "run frontend server", "start dev server", "next dev"))
        add("save_memory", "Save Memory", ("remember", "rember", "remeber", "save memory", "add memory", "memorize", "memo"))
        add("generate_study_artifact", "Generate Notes", ("generate notes", "create notes", "make notes", "notes"), {"artifact_type": "notes"})
        add("generate_study_artifact", "Generate Flashcards", ("generate flashcards", "create flashcards", "make flashcards", "flashcards", "flashcrds"), {"artifact_type": "flashcards"})
        add("generate_study_artifact", "Generate Quiz", ("generate quiz", "create quiz", "make quiz", "quiz", "quizz"), {"artifact_type": "quiz"})
        add("generate_study_artifact", "Generate Revision Plan", ("generate revision plan", "create revision plan", "make revision plan", "revision plan"), {"artifact_type": "revision_plan"})
        add("generate_study_artifact", "Generate Viva Questions", ("generate viva questions", "create viva questions", "make viva questions", "viva questions", "viva"), {"artifact_type": "viva_questions"})
        add(
            "generate_mock_test",
            "Generate Mock Test",
            (
                "create mock test",
                "generate mock test",
                "make mock test",
                "mock test",
                "mock exam",
                "practice test",
                "create practice test",
                "generate practice test",
                "make test",
                "create test",
            ),
            {"question_count": 10, "difficulty": "mixed", "mode": "mcq", "duration_minutes": 20},
        )
        return candidates

    def _llm_candidates(self) -> list[dict[str, Any]]:
        return [
            {
                "command_id": candidate.command_id,
                "label": candidate.label,
                "aliases": list(candidate.aliases[:12]),
                "params": candidate.params,
            }
            for candidate in self.intent_candidates
            if candidate.command_id != "blocked_action"
        ]

    def _parse_llm_json(self, raw: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except Exception:
                return None
        return payload if isinstance(payload, dict) else None

    def _safe_target_map(self) -> dict[str, DesktopTarget]:
        names = {
            "youtube": "YouTube",
            "google": "Google",
            "gmail": "Gmail",
            "google_docs": "Google Docs",
            "google_drive": "Google Drive",
            "calendar": "Google Calendar",
            "notepad": "Notepad",
            "calculator": "Calculator",
            "file_explorer": "File Explorer",
        }
        targets: dict[str, DesktopTarget] = {}
        for key, label in names.items():
            match = next((target for target in SAFE_TARGETS if target.label == label), None)
            if match:
                targets[key] = match
        return targets

    def _detect_allowed_target(self, normalized: str) -> str | None:
        for key, target in self.safe_targets.items():
            if any(alias in normalized for alias in target.aliases):
                return key
        return None

    def _provider_status(self) -> dict[str, bool]:
        nvidia_configured = getattr(self.llm, "provider_configured", lambda provider: False)("nvidia")
        return {
            "cerebras": self.settings.has_cerebras,
            "nvidia": nvidia_configured,
            "tavily": self.settings.has_tavily,
            "openalex": True,
            "semantic_scholar": True,
            "duckduckgo_fallback": True,
            "piper_local": self.voice.status().enabled,
        }

    def _report_summary(self, report: Any) -> dict[str, Any]:
        return {
            "id": report.id,
            "title": report.title,
            "created_at": report.created_at.isoformat(),
            "download_url": report.download_url,
        }

    def _artifact_summary(self, artifact: Any) -> dict[str, Any]:
        return {
            "id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "title": artifact.title,
            "source": artifact.source,
            "created_at": artifact.created_at.isoformat(),
            "markdown": artifact.markdown[:1200],
        }

    def _select_mock_test(self, tests: list[Any], topic: str) -> Any | None:
        if not tests:
            return None
        clean_tests = [test for test in tests if not self._mock_topic_looks_like_status(str(getattr(test, "topic", "")))]
        tests = clean_tests or tests
        topic = self._normalize(topic)
        if not topic:
            return tests[0] if clean_tests else None
        topic_compact = self._compact(topic)
        best: tuple[int, int, Any] | None = None
        for test in tests:
            test_topic = self._normalize(str(getattr(test, "topic", "")))
            test_compact = self._compact(test_topic)
            if test_compact == topic_compact:
                score = 120
            else:
                score = self._alias_score(test_topic, topic)
                if len(topic_compact) <= 4 and topic_compact and topic_compact in test_compact:
                    score = min(score, 78)
                score -= min(20, max(0, len(test_compact) - len(topic_compact)) // 3)
            length_delta = abs(len(test_compact) - len(topic_compact))
            if best is None or score > best[0] or (score == best[0] and length_delta < best[1]):
                best = (score, length_delta, test)
        if best and best[0] >= 68:
            return best[2]
        return None

    async def _url_responds(self, url: str) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(url)
            return response.status_code < 500
        except Exception:
            return False

    def _extract_direct_url(self, text: str) -> str | None:
        match = re.search(r"https?://[^\s]+", text.strip(), re.IGNORECASE)
        if not match:
            return None
        candidate = match.group(0).rstrip(".,)")
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return candidate
        return None

    def _infer_memory_category(self, text: str) -> str:
        normalized = self._normalize(text)
        if re.search(r"\b(course|class|exam|semester|subject)\b", normalized):
            return "course"
        if re.search(r"\b(project|repo|app|astra)\b", normalized):
            return "project"
        if re.search(r"\b(goal|target|deadline)\b", normalized):
            return "goal"
        if re.search(r"\b(prefer|style|tone|format)\b", normalized):
            return "preference"
        return "general"

    def _infer_study_type(self, normalized: str) -> str | None:
        if not re.search(r"\b(generate|create|make|build)\b", normalized):
            return None
        if "flashcard" in normalized:
            return "flashcards"
        if "quiz" in normalized:
            return "quiz"
        if "revision plan" in normalized:
            return "revision_plan"
        if "viva" in normalized:
            return "viva_questions"
        if "notes" in normalized:
            return "notes"
        return None

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:/._-]+", " ", text.lower())).strip()
