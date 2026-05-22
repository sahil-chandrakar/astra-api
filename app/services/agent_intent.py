from __future__ import annotations

import json
import re
from typing import Any

from app.config import Settings
from app.models import AgentCommandRequest
from app.services.llm import LlmService


class AgentIntentParser:
    """Semantic parser for safe registry intents.

    The parser may use the configured LLM, but it only emits registry command
    IDs and plain parameters. The safe registry remains the execution gate.
    """

    def __init__(self, settings: Settings, llm: LlmService):
        self.settings = settings
        self.llm = llm

    async def resolve(
        self,
        text: str,
        normalized: str,
        confirmed: bool,
        allowed_command_ids: set[str],
    ) -> AgentCommandRequest | None:
        if not self._can_be_semantic_agent_intent(normalized):
            return None
        if self.llm.model_configured():
            llm_request = await self._resolve_with_llm(text, normalized, confirmed, allowed_command_ids)
            if llm_request:
                return llm_request
        return self._resolve_locally(text, normalized, confirmed)

    async def _resolve_with_llm(
        self,
        text: str,
        normalized: str,
        confirmed: bool,
        allowed_command_ids: set[str],
    ) -> AgentCommandRequest | None:
        system_prompt = (
            "You are Astra's semantic task router for Agent Mode. Return strict JSON only. "
            "Understand the user's meaning first, then choose at most one allowed command. "
            "You may classify intent and extract safe parameters, but you must not invent commands, apps, shell strings, URLs, or actions. "
            "For learning prompts such as 'test me on...', 'quiz me about...', 'ask me questions on...', or 'practice <topic>', choose generate_mock_test because the user wants a new assessment. "
            "For saved-item prompts such as 'open my latest mock test', 'where is my mock test', or 'mock test not created', choose open_latest_mock_test because the user wants an existing test/status. "
            "If the user asks for PYQ, previous-year, past-paper, or official questions, set requires_sources true and source_requirement pyq_required. "
            "If the user says PYQ-style, previous-year style, or source-free practice, set source_requirement none and add a PYQ-style practice constraint. "
            "Do not convert UGC NET CS/CSE/Computer Science to Python. "
            "For vague source requests like 'ugc net pyq' with no create/generate/mock/test/question action, choose clarify_agent_intent."
        )
        user_prompt = json.dumps(
            {
                "prompt": text,
                "normalized_prompt": normalized,
                "allowed_command_ids": sorted(allowed_command_ids | {"clarify_agent_intent"}),
                "output_schema": {
                    "intent": "generate_mock_test|open_latest_mock_test|list_mock_tests|open_allowlisted_target|clarify|chat|null",
                    "command_id": "allowed command id or null",
                    "topic": "string",
                    "exam": "string",
                    "subject": "string",
                    "constraints": ["string"],
                    "question_count": "integer",
                    "difficulty": "easy|medium|hard|mixed",
                    "mode": "mcq",
                    "duration_minutes": "integer",
                    "requires_sources": "boolean",
                    "source_requirement": "none|pyq_required|source_backed",
                    "source_mode": "uploaded_docs",
                    "source_query": "string",
                    "target": "approved target key for open_allowlisted_target, such as calculator",
                    "params": {"target": "approved target key"},
                    "is_new_creation": "boolean",
                    "is_existing_item_request": "boolean",
                    "needs_clarification": "boolean",
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
        payload = self._parse_json(raw)
        if not payload:
            return None

        command_id = str(payload.get("command_id") or "").strip()
        if not command_id and payload.get("needs_clarification"):
            command_id = "clarify_agent_intent"
        if command_id.lower() in {"chat", "null", "none"}:
            return None
        if not command_id:
            return None

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.68 and command_id in allowed_command_ids:
            return None

        is_new_creation = bool(payload.get("is_new_creation"))
        is_existing_item_request = bool(payload.get("is_existing_item_request"))

        if command_id == "generate_mock_test":
            if is_existing_item_request and not is_new_creation:
                return self._open_latest_request(text, normalized, confirmed, payload, confidence)
            if not is_new_creation and not self._is_mock_generation_intent(normalized):
                if self._is_mock_existing_intent(normalized):
                    return self._open_latest_request(text, normalized, confirmed, payload, confidence)
                if self._mentions_pyq(normalized):
                    return self._clarify_pyq_request(text, normalized, confirmed)
                return None

        if command_id == "open_latest_mock_test" and is_new_creation and not is_existing_item_request:
            params = self._mock_params(text, normalized, payload)
            resolution = self._resolution("llm", confidence, str(payload.get("matched_alias") or payload.get("reason") or "new assessment"), normalized)
            resolution["intent"] = "generate_mock_test"
            resolution["reason"] = str(payload.get("reason") or "")[:180]
            resolution["semantic_override"] = "open_latest_to_generate"
            return AgentCommandRequest(command_id="generate_mock_test", input_text=text, params=params, confirmed=confirmed, resolution=resolution)

        if command_id == "generate_mock_test" and not self._is_mock_generation_intent(normalized):
            if self._mentions_pyq(normalized):
                return self._clarify_pyq_request(text, normalized, confirmed)

        if command_id == "clarify_agent_intent" or payload.get("needs_clarification"):
            if not self._mentions_pyq(normalized):
                return None
            return self._clarify_pyq_request(text, normalized, confirmed, payload, confidence)

        if command_id == "open_latest_mock_test":
            return self._open_latest_request(text, normalized, confirmed, payload, confidence)

        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        if command_id == "generate_mock_test":
            params = self._mock_params(text, normalized, payload)
        elif command_id == "open_allowlisted_target" and payload.get("target") and "target" not in params:
            params = {**params, "target": str(payload.get("target") or "").strip()}

        resolution = self._resolution("llm", confidence, str(payload.get("matched_alias") or payload.get("reason") or command_id), normalized)
        resolution["intent"] = str(payload.get("intent") or command_id)[:80]
        resolution["reason"] = str(payload.get("reason") or "")[:180]
        return AgentCommandRequest(command_id=command_id, input_text=text, params=params, confirmed=confirmed, resolution=resolution)

    def _resolve_locally(self, text: str, normalized: str, confirmed: bool) -> AgentCommandRequest | None:
        if self._is_mock_generation_intent(normalized):
            params = self._mock_params(text, normalized, {})
            resolution = self._resolution("semantic_local", 0.96, "new assessment intent", normalized)
            resolution["intent"] = "generate_mock_test"
            return AgentCommandRequest(
                command_id="generate_mock_test",
                input_text=text,
                params=params,
                confirmed=confirmed,
                resolution=resolution,
            )
        if self._is_mock_existing_intent(normalized):
            return self._open_latest_request(text, normalized, confirmed, {}, 1.0)
        if self._mentions_pyq(normalized) and not self._is_mock_generation_intent(normalized):
            return self._clarify_pyq_request(text, normalized, confirmed)
        return None

    def _can_be_semantic_agent_intent(self, normalized: str) -> bool:
        if self._looks_like_small_talk(normalized):
            return False
        return bool(normalized)

    def _looks_like_small_talk(self, normalized: str) -> bool:
        compact = re.sub(r"[^a-z0-9]+", "", normalized.lower())
        greetings = {
            "hi",
            "hii",
            "hiii",
            "hello",
            "helo",
            "hey",
            "heyy",
            "hallo",
            "thanks",
            "thankyou",
            "ok",
            "okay",
            "yo",
        }
        if compact in greetings:
            return True
        if re.search(r"\b(what is your name|what's your name|who are you|your name|how are you|how r you|how are u)\b", normalized):
            return True
        return bool(re.fullmatch(r"(hi+|he+y+|hello+|hlo+|hii+)", compact))

    def _mock_params(self, text: str, normalized: str, payload: dict[str, Any]) -> dict[str, Any]:
        topic = self._canonical_topic(text, normalized, payload)
        exam, subject = self._exam_subject(topic, normalized, payload)
        source_requirement = str(payload.get("source_requirement") or "").strip()
        pyq_style = self._mentions_pyq_style(normalized, payload)
        requires_sources = (bool(payload.get("requires_sources")) or self._mentions_pyq(normalized)) and not pyq_style
        if source_requirement not in {"none", "pyq_required", "source_backed"}:
            source_requirement = "pyq_required" if requires_sources else "none"
        if pyq_style:
            source_requirement = "none"

        source_mode = "uploaded_docs"

        constraints = payload.get("constraints") if isinstance(payload.get("constraints"), list) else []
        clean_constraints = [str(item).strip()[:80] for item in constraints if str(item).strip()]
        if requires_sources and not any("pyq" in item.lower() for item in clean_constraints):
            clean_constraints.append("PYQ required")
        if pyq_style and not any("pyq-style" in item.lower() for item in clean_constraints):
            clean_constraints.append("PYQ-style practice")

        return {
            "topic": topic,
            "exam": exam,
            "subject": subject,
            "question_count": self._int_param(payload.get("question_count"), self._question_count_from_text(normalized), 10, 1, 50),
            "difficulty": self._difficulty(payload.get("difficulty")),
            "mode": "mcq",
            "duration_minutes": self._int_param(payload.get("duration_minutes"), self._duration_from_text(normalized), 20, 1, 180),
            "source_requirement": source_requirement,
            "source_mode": source_mode,
            "source_query": str(payload.get("source_query") or self._source_query(topic, requires_sources)).strip()[:240],
            "constraints": clean_constraints[:8],
        }

    def _open_latest_request(
        self,
        text: str,
        normalized: str,
        confirmed: bool,
        payload: dict[str, Any],
        confidence: float,
    ) -> AgentCommandRequest:
        topic = str(payload.get("topic") or self._existing_mock_topic(text)).strip(" .")
        resolution = self._resolution("llm" if payload else "semantic_local", confidence, "mock test follow-up", normalized)
        resolution["intent"] = "open_existing_tool"
        return AgentCommandRequest(
            command_id="open_latest_mock_test",
            input_text=text,
            params={"topic": topic} if topic else {},
            confirmed=confirmed,
            resolution=resolution,
        )

    def _clarify_pyq_request(
        self,
        text: str,
        normalized: str,
        confirmed: bool,
        payload: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> AgentCommandRequest:
        topic = self._canonical_topic(text, normalized, payload or {})
        message = (
            f"For {topic} PYQ, I need source material before creating a mock test. "
            "Upload a PYQ PDF/source document, or say PYQ-style practice if you want AI-created practice."
        )
        resolution = self._resolution("llm" if payload else "semantic_local", confidence, "pyq clarification", normalized)
        resolution["intent"] = "clarify_source_required"
        return AgentCommandRequest(
            command_id="clarify_agent_intent",
            input_text=text,
            params={
                "message": message,
                "topic": topic,
                "actions": ["Use uploaded PDF", "Generate PYQ-style practice"],
            },
            confirmed=confirmed,
            resolution=resolution,
        )

    def _canonical_topic(self, text: str, normalized: str, payload: dict[str, Any]) -> str:
        raw_topic = str(payload.get("topic") or "").strip()
        raw_exam = str(payload.get("exam") or "").strip()
        raw_subject = str(payload.get("subject") or "").strip()
        candidate = self._dedupe_words(raw_topic) if raw_topic else self._dedupe_topic_parts(raw_exam, raw_subject)
        candidate = candidate or self._extract_topic_from_text(text)
        candidate_normalized = self._normalize(candidate) or normalized

        if self._mentions_ugc_net(candidate_normalized) or self._mentions_ugc_net(normalized):
            if re.search(r"\b(cs|cse|computer science|computer applications?)\b", candidate_normalized) or re.search(
                r"\b(cs|cse|computer science|computer applications?)\b", normalized
            ):
                return "UGC NET Computer Science"
            return "UGC NET"
        return re.sub(r"\s+", " ", candidate).strip(" .")[:120] or "general aptitude"

    def _dedupe_topic_parts(self, *parts: str) -> str:
        clean_parts: list[str] = []
        seen: set[str] = set()
        seen_tokens: set[str] = set()
        for part in parts:
            clean = re.sub(r"\s+", " ", str(part or "")).strip(" .")
            if not clean:
                continue
            clean = self._dedupe_words(clean)
            key = self._normalize(clean)
            if not key:
                continue
            tokens = set(key.split())
            if key in seen:
                continue
            if tokens and tokens.issubset(seen_tokens):
                continue
            if clean_parts and tokens & seen_tokens:
                novel_words = [word for word in clean.split() if self._normalize(word) not in seen_tokens]
                clean = " ".join(novel_words).strip(" .")
                key = self._normalize(clean)
                tokens = set(key.split())
            if not clean or not key or (tokens and tokens.issubset(seen_tokens)):
                continue
            clean_parts.append(clean)
            seen.add(key)
            seen_tokens.update(tokens)
        return " ".join(clean_parts).strip()

    def _dedupe_words(self, text: str) -> str:
        words: list[str] = []
        seen: set[str] = set()
        for word in text.split():
            key = self._normalize(word)
            if key and key in seen:
                continue
            words.append(word)
            if key:
                seen.add(key)
        return " ".join(words).strip()

    def _exam_subject(self, topic: str, normalized: str, payload: dict[str, Any]) -> tuple[str, str]:
        exam = str(payload.get("exam") or "").strip()
        subject = str(payload.get("subject") or "").strip()
        topic_normalized = self._normalize(topic)
        combined = f"{normalized} {topic_normalized}"
        if self._mentions_ugc_net(combined):
            return "UGC NET", subject or (
                "Computer Science"
                if re.search(r"\b(cs|cse|computer science|computer applications?)\b", combined)
                else "Paper I: Teaching and Research Aptitude"
            )
        if re.search(r"\bgate\b", combined):
            return "GATE", subject or ("Computer Science" if re.search(r"\b(cs|cse|computer science|dbms|os|toc|cn)\b", combined) else "")
        if re.search(r"\bjee\b", combined):
            subject = subject or self._subject_from_terms(combined, {"physics": "Physics", "chemistry": "Chemistry", "math": "Mathematics", "mathematics": "Mathematics"})
            return "JEE", subject
        if re.search(r"\bneet\b", combined):
            subject = subject or self._subject_from_terms(combined, {"biology": "Biology", "bio": "Biology", "genetics": "Biology", "physics": "Physics", "chemistry": "Chemistry"})
            return "NEET", subject
        if re.search(r"\b(upsc|civil services)\b", combined):
            return "UPSC", subject or "General Studies"
        if re.search(r"\b(ssc|banking|cgl)\b", combined):
            return "SSC/Banking", subject or "Quantitative Aptitude and Reasoning"
        if re.search(r"\bcurrent affairs?\b", combined):
            return exam[:80], subject or "Current Affairs"
        return exam[:80], subject[:80]

    def _subject_from_terms(self, normalized: str, mapping: dict[str, str]) -> str:
        for term, label in mapping.items():
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                return label
        return ""

    def _extract_topic_from_text(self, text: str) -> str:
        clean = re.sub(
            r"^\s*(please\s+)?(create|generate|make|build|prepare|new|another)\s+(a\s+|an\s+)?",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"^\s*(please\s+)?(test|quiz|assess|challenge)\s+(me|my knowledge)?\s*(on|about|in|for)?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"^\s*(please\s+)?(ask|give)\s+(me\s+)?(some\s+|a\s+)?(questions?|mcqs?|quiz|test)\s*(on|about|in|for)?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\b(mock\s+test|mock\s+exam|practice\s+test|test|quiz|questions?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(current affairs?)\s+(on|about|for)\s+(the\s+)?topic\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(current affairs?)\s+(on|about|for)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(on|about|for)\s+(the\s+)?topic\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^\s*(for|on|about|of)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^\s*(another|new)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(only|based\s+on)?\s*(pyq|previous\s+year|past\s+paper|official)\s+(questions?|papers?)?\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(questions?\s+should\s+be\s+there|should\s+be\s+there|there\s+should\s+be)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip(" .,")

    def _existing_mock_topic(self, text: str) -> str:
        clean = re.sub(r"\b(mock\s+test|mock|test|where|show|open|view|see|find|latest|not\s+created|till\s+now|yet|ready|is|my|the)\b", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", clean).strip(" .")

    def _source_query(self, topic: str, requires_sources: bool) -> str:
        if not requires_sources:
            return topic
        return f"{topic} previous year questions PYQ MCQ"

    def _mentions_ugc_net(self, normalized: str) -> bool:
        return bool(("ugc" in normalized and "net" in normalized) or "net ugc" in normalized)

    def _mentions_pyq(self, normalized: str) -> bool:
        return bool(re.search(r"\b(pyq|previous\s+year|past\s+paper|past\s+papers|official\s+questions?|year\s+questions?)\b", normalized))

    def _mentions_mock_test(self, normalized: str) -> bool:
        return bool(re.search(r"\b(mock|mocktest|practice\s+test|mock\s+exam|quiz|test|questions?)\b", normalized))

    def _mentions_pyq_style(self, normalized: str, payload: dict[str, Any] | None = None) -> bool:
        combined = normalized
        if payload:
            constraints = payload.get("constraints") if isinstance(payload.get("constraints"), list) else []
            combined = f"{combined} {' '.join(str(item) for item in constraints)} {payload.get('source_requirement', '')}"
        return bool(re.search(r"\b(pyq\s*style|pyq-style|previous\s+year\s+style|past\s+paper\s+style)\b", combined, flags=re.IGNORECASE))

    def _is_mock_generation_intent(self, normalized: str) -> bool:
        if re.search(r"\b(not created|not generated|did not create|didnt create|not showing|till now|where|show|open|view|see|find)\b", normalized):
            return False
        if self._is_assessment_generation_intent(normalized):
            return True
        has_action = bool(re.search(r"\b(create|generate|make|build|prepare|new|another|creat|genrate|generte|mak|preprare)\b", normalized))
        has_object = bool(re.search(r"\b(mock|exam|practice|test|quiz|questions?)\b", normalized))
        return has_action and has_object and not re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized)

    def _is_assessment_generation_intent(self, normalized: str) -> bool:
        if re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized):
            return False
        if re.search(r"\b(test|quiz|assess|challenge)\s+(me|my knowledge)?\b", normalized):
            return True
        if re.search(r"\b(ask|give)\s+(me\s+)?(some\s+|a\s+)?(questions?|mcqs?|quiz)\b", normalized):
            return True
        if re.search(r"\bpractice\s+(on|with|for|about)?\b", normalized) and not self._is_mock_existing_intent(normalized):
            return True
        return False

    def _is_mock_existing_intent(self, normalized: str) -> bool:
        mentions_mock = bool(re.search(r"\b(mock|mocktest|test)\b", normalized))
        if not mentions_mock:
            return False
        if self._is_mock_generation_intent(normalized):
            return False
        return bool(re.search(r"\b(where|show|open|view|see|find|latest|created|generated|available|missing|not created|not generated|till now|yet|ready)\b", normalized))

    def _question_count_from_text(self, normalized: str) -> int | None:
        match = re.search(r"\b(\d{1,2})\s*(?:questions?|q|mcqs?)\b", normalized)
        return int(match.group(1)) if match else None

    def _duration_from_text(self, normalized: str) -> int | None:
        match = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?|min)\b", normalized)
        return int(match.group(1)) if match else None

    def _difficulty(self, value: Any) -> str:
        difficulty = str(value or "mixed").strip().lower()
        return difficulty if difficulty in {"easy", "medium", "hard", "mixed"} else "mixed"

    def _int_param(self, primary: Any, fallback: int | None, default: int, minimum: int, maximum: int) -> int:
        value = primary if primary not in {None, ""} else fallback if fallback is not None else default
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(maximum, number))

    def _resolution(self, source: str, confidence: float, matched_alias: str, normalized: str) -> dict[str, Any]:
        return {
            "source": source,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "matched_alias": matched_alias,
            "normalized_prompt": normalized,
        }

    def _parse_json(self, raw: str | None) -> dict[str, Any] | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
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

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:/._-]+", " ", text.lower())).strip()
