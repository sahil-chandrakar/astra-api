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
        if command_id not in allowed_command_ids and command_id != "clarify_agent_intent":
            intent = str(payload.get("intent") or "").strip().lower()
            if intent == "generate_mock_test" or self._is_mock_generation_intent(normalized):
                command_id = "generate_mock_test"
            elif intent == "open_latest_mock_test" or self._is_mock_existing_intent(normalized):
                command_id = "open_latest_mock_test"
            elif intent == "list_mock_tests" and "list_mock_tests" in allowed_command_ids:
                command_id = "list_mock_tests"
            else:
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
        topic = self._topic_focus(topic, exam, subject)
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
        if pyq_style:
            clean_constraints = [
                item
                for item in clean_constraints
                if not re.search(r"\b(?:real\s+)?pyq[-_ ]?styles?\b", item, flags=re.IGNORECASE)
            ]
        if re.search(r"\b(numerical|numeric|calculation|calculate|quantitative|problem[- ]?solving)\b", normalized):
            if not any("numerical" in item.lower() or "numeric" in item.lower() for item in clean_constraints):
                clean_constraints.append("numerical questions")
        if requires_sources and not any("pyq" in item.lower() for item in clean_constraints):
            clean_constraints.append("PYQ required")
        if pyq_style and not any("pyq-style" in item.lower() for item in clean_constraints):
            clean_constraints.append("PYQ-style practice")

        source_query = self._clean_source_query(str(payload.get("source_query") or ""), topic, requires_sources)
        difficulty_from_text = self._difficulty_from_text(normalized)
        duration_from_text = self._duration_from_text(normalized)
        question_count_from_text = self._question_count_from_text(normalized)
        return {
            "topic": topic,
            "exam": exam,
            "subject": subject,
            "question_count": self._int_param(question_count_from_text, None, 10, 1, 50),
            "difficulty": difficulty_from_text or self._difficulty(payload.get("difficulty")),
            "mode": "mcq",
            "duration_minutes": self._int_param(duration_from_text, None, 20, 1, 180),
            "source_requirement": source_requirement,
            "source_mode": source_mode,
            "source_query": source_query,
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
        candidate = self._clean_mock_topic(candidate, normalized) if candidate else ""
        candidate = candidate or self._extract_topic_from_text(text)
        candidate_normalized = self._normalize(candidate) or normalized

        if self._mentions_ugc_net(candidate_normalized) or self._mentions_ugc_net(normalized):
            combined = f"{candidate_normalized} {normalized}"
            focus_source = " ".join(part for part in [candidate, text] if part).strip()
            subject_hint = self._ugc_net_subject_hint(combined)
            if self._is_ugc_net_cs_context(combined):
                focus = self._ugc_net_topic_focus(focus_source, normalized, "Computer Science")
                return self._dedupe_topic_parts("UGC NET Computer Science", focus) if focus else "UGC NET Computer Science"
            if subject_hint:
                focus = self._ugc_net_topic_focus(focus_source, normalized, subject_hint)
                return self._dedupe_topic_parts(f"UGC NET {subject_hint}", focus) if focus else f"UGC NET {subject_hint}"
            if self._is_ugc_net_paper_two_context(combined):
                focus = self._ugc_net_topic_focus(focus_source, normalized, "Paper II")
                return self._dedupe_topic_parts("UGC NET Paper II", focus) if focus else "UGC NET Paper II"
            if self._is_ugc_net_paper_one_context(combined):
                focus = self._ugc_net_topic_focus(focus_source, normalized, "Paper I")
                return self._dedupe_topic_parts("UGC NET", focus) if focus else "UGC NET"
            return "UGC NET"
        return re.sub(r"\s+", " ", candidate).strip(" .")[:120] or "general aptitude"

    def _clean_mock_topic(self, text: str, normalized: str = "") -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip(" .,")
        if not clean:
            return ""
        clean = re.sub(
            r"^\s*(please\s+)?(create|generate|make|build|prepare|new|another|creat|genrate|generte|mak|preprare)\s+(me\s+)?(a\s+|an\s+|another\s+|new\s+)?",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(
            r"^\s*(mock\s+test|mock\s+exam|practice\s+test|test|quiz|exam|questions?|mcqs?)\s*(for|on|about|of|in)?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\b(?:with|having|containing)\s+\d{1,2}\s*[- ]?(?:questions?|qs?|mcqs?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:with|having|containing)?\s*\d{1,2}\s+(?:easy|medium|hard|mixed|tough|advanced|basic)\s+(?:questions?|qs?|mcqs?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b\d{1,2}\s*[- ]?(?:questions?|qs?|mcqs?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|fifty)\s+(?:questions?|qs?|mcqs?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:mcqs?|multiple\s+choice\s+questions?|questions?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:with|for|of)?\s*\d{1,3}\s*(?:minutes?|mins?|min|hours?|hrs?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:easy|medium|hard|mixed)\s+(?:difficulty|level)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:difficulty|level)\s*[:=-]?\s*(?:easy|medium|hard|mixed)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:easy|medium|hard|mixed|tough|toughest|advanced|challenging|beginner|basic|simple)\b\s*$", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:numerical|numeric|calculation[- ]?based|quantitative|problem[- ]?solving)\b\s*$", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:difficulty|duration|timer|timed|minutes?|mins?|mcq|mock|test|quiz)\b\s*$", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:real\s+)?(?:pyq|previous\s+year|past\s+paper)\s*[- ]?styles?\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:pyq|previous\s+year|past\s+paper|official)\s+(?:questions?|papers?|styles?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(current affairs?)\s+(on|about|for)\s+(the\s+)?topic\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(current affairs?)\s+(on|about|for)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(on|about|for)\s+(the\s+)?topic\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^\s*(for|on|about|of|in)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+(with|for|and|on|about|of|in)\s*$", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|fifty|\d{1,2})\s*$", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+(with|for|and|on|about|of|in)\s*$", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*[,;]\s*", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip(" .,")
        if not clean and normalized:
            clean = self._extract_topic_from_text(normalized)
        return self._dedupe_words(clean)

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
            if self._is_ugc_net_cs_context(combined):
                return "UGC NET", "Computer Science"
            subject_hint = self._ugc_net_subject_hint(combined)
            if subject_hint:
                return "UGC NET", subject_hint
            if subject and not self._subject_is_generic_ugc_paper_one(subject):
                return "UGC NET", subject[:80]
            if self._is_ugc_net_paper_two_context(combined):
                return "UGC NET", "Paper II"
            return "UGC NET", subject or "Paper I: Teaching and Research Aptitude"
        if re.search(r"\bgate\b", combined):
            return "GATE", subject or (
                "Computer Science"
                if re.search(
                    r"\b(cs|cse|computer science|dbms|database|os|operating|deadlock|toc|automata|compiler|cn|network|subnet|algorithm|data structures?)\b",
                    combined,
                )
                else ""
            )
        if re.search(r"\bjee\b", combined):
            subject = subject or self._subject_from_terms(
                combined,
                {"physics": "Physics", "phyics": "Physics", "chemistry": "Chemistry", "math": "Mathematics", "mathematics": "Mathematics"},
            )
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
        class_match = re.search(r"\b(?:class|grade|standard|std)\s*(?P<class>\d{1,2})\b", combined)
        if class_match:
            subject = subject or self._subject_from_terms(
                combined,
                {
                    "physics": "Physics",
                    "phyics": "Physics",
                    "chemistry": "Chemistry",
                    "biology": "Biology",
                    "science": "Science",
                    "math": "Mathematics",
                    "maths": "Mathematics",
                    "mathematics": "Mathematics",
                    "history": "History",
                    "geography": "Geography",
                    "english": "English",
                    "computer science": "Computer Science",
                },
            )
            return f"Class {class_match.group('class')}", subject
        subject = subject or self._subject_from_terms(
            combined,
            {
                "python": "Python Programming",
                "dsa": "Data Structures and Algorithms",
                "data structures": "Data Structures and Algorithms",
                "algorithms": "Data Structures and Algorithms",
                "dbms": "DBMS",
                "database": "DBMS",
                "operating system": "Operating Systems",
                "deadlock": "Operating Systems",
                "computer networks": "Computer Networks",
                "networking": "Computer Networks",
                "cyber security": "Cyber Security",
                "discrete mathematics": "Discrete Mathematics",
                "indian history": "Indian History",
            },
        )
        return exam[:80], subject[:80]

    def _topic_focus(self, topic: str, exam: str, subject: str) -> str:
        clean = topic
        normalized_subject = self._normalize(subject)
        removals = [
            r"\b(?:class|grade|standard|std)\s*\d{1,2}\b",
            r"\b(?:jee|neet|gate|upsc|ssc|banking|cgl|school|ugc|net|nta)\b",
            r"\bpaper\s*(?:i|ii|1|2)\b",
            r"\bpaper[- ]?(?:i|ii|1|2)\b",
            r"\b(?:cse|cs)\b",
        ]
        if self._normalize(exam) == "jee":
            removals.append(r"\b(?:mains?|advanced|iit)\b")
        subject_patterns = {
            "physics": r"\b(?:physics|phyics)\b",
            "chemistry": r"\bchemistry\b",
            "biology": r"\b(?:biology|bio)\b",
            "mathematics": r"\b(?:math|maths|mathematics)\b",
            "computer science": r"\bcomputer\s+science\b",
            "computer applications": r"\bcomputer\s+applications?\b",
            "science": r"\bscience\b",
            "current affairs": r"\bcurrent\s+affairs?\b",
            "history": r"\bhistory\b",
            "geography": r"\bgeography\b",
            "economics": r"\beconomics?\b",
            "commerce": r"\bcommerce\b",
            "management": r"\bmanagement\b",
            "education": r"\beducation\b",
            "english": r"\benglish\b",
            "hindi": r"\bhindi\b",
            "sociology": r"\bsociology\b",
            "psychology": r"\bpsychology\b",
            "law": r"\blaw\b",
            "political science": r"\b(?:political\s+science|political)\b",
        }
        for key, pattern in subject_patterns.items():
            if key and key in normalized_subject:
                removals.append(pattern)
        for pattern in removals:
            clean = re.sub(pattern, " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^\s*(for|on|about|of|in)\s+", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+(for|on|about|of|in)\s*$", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:easy|medium|hard|mixed|tough|toughest|advanced|challenging|beginner|basic|simple)\b\s*$", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:numerical|numeric|calculation[- ]?based|quantitative|problem[- ]?solving)\b\s*$", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean).strip(" .,")
        if len(clean) >= 3:
            return self._dedupe_words(clean)[:120]
        return topic[:120]

    def _subject_from_terms(self, normalized: str, mapping: dict[str, str]) -> str:
        for term, label in mapping.items():
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                return label
        return ""

    def _subject_is_generic_ugc_paper_one(self, subject: str) -> bool:
        normalized = self._normalize(subject)
        return bool(re.search(r"\bpaper\s*i\b|\bpaper\s*1\b|teaching\s+and\s+research\s+aptitude", normalized))

    def _is_ugc_net_paper_one_context(self, normalized: str) -> bool:
        return self._mentions_ugc_net(normalized) and bool(re.search(r"\b(?:paper|papr|ppr)\s*(?:1|i)\b|\bpaper[- ]?(?:1|i)\b|teaching\s+aptitude|research\s+aptitude", normalized))

    def _is_ugc_net_paper_two_context(self, normalized: str) -> bool:
        return self._mentions_ugc_net(normalized) and bool(re.search(r"\b(?:paper|papr|ppr)\s*(?:2|ii)\b|\bpaper[- ]?(?:2|ii)\b", normalized))

    def _is_ugc_net_cs_context(self, normalized: str) -> bool:
        if not self._mentions_ugc_net(normalized):
            return False
        return bool(
            re.search(
                r"\b(?:cs|cse|computer\s+science|computer\s+applications?|dsa|data\s+structures?|data\s+structure\s+and\s+algorithms?|"
                r"algorithms?|arrays?|dbms|database|operating\s+systems?|computer\s+networks?|toc|automata|compiler|software\s+engineering|"
                r"artificial\s+intelligence|machine\s+learning|discrete\s+mathematics|boolean|stack|queue|tree|graph|heap|hashing)\b",
                normalized,
            )
        )

    def _ugc_net_subject_hint(self, normalized: str) -> str:
        subject_patterns = (
            (r"\b(?:cs|cse|computer\s+science|computer\s+applications?|dsa|data\s+structures?|data\s+structure\s+and\s+algorithms?|dbms|operating\s+systems?|computer\s+networks?|toc|automata|compiler)\b", "Computer Science"),
            (r"\bpolitical\s+science\b", "Political Science"),
            (r"\blibrary\s+(?:and\s+information\s+)?science\b", "Library and Information Science"),
            (r"\benvironmental\s+science\b", "Environmental Science"),
            (r"\belectronic\s+science\b", "Electronic Science"),
            (r"\bphysical\s+education\b", "Physical Education"),
            (r"\bpublic\s+administration\b", "Public Administration"),
            (r"\bsocial\s+work\b", "Social Work"),
            (r"\bhome\s+science\b", "Home Science"),
            (r"\b(?:math|maths|mathematics)\b", "Mathematics"),
            (r"\bcommerce\b", "Commerce"),
            (r"\bmanagement\b", "Management"),
            (r"\beconomics?\b", "Economics"),
            (r"\beducation\b", "Education"),
            (r"\benglish\b", "English"),
            (r"\bhindi\b", "Hindi"),
            (r"\bhistory\b", "History"),
            (r"\bgeography\b", "Geography"),
            (r"\bsociology\b", "Sociology"),
            (r"\bpsychology\b", "Psychology"),
            (r"\blaw\b", "Law"),
            (r"\bphilosophy\b", "Philosophy"),
            (r"\bsanskrit\b", "Sanskrit"),
            (r"\burdu\b", "Urdu"),
        )
        for pattern, subject in subject_patterns:
            if re.search(pattern, normalized):
                return subject
        return ""

    def _ugc_net_topic_focus(self, text: str, normalized: str, subject: str) -> str:
        clean = self._clean_mock_topic(text, normalized)
        clean = re.sub(r"\([^)]*\bpaper\s*(?:i|ii|1|2)\b[^)]*\)", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:ugc|net|nta)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:paper|papr|ppr)\s*(?:i|ii|1|2)\b|\bpaper[- ]?(?:i|ii|1|2)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:real\s+)?(?:pyq|previous\s+year|past\s+paper)\s*[- ]?(?:styles?|stiles?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:only|real|actual|official|source[- ]?backed|based)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(?:pyq|previous\s+year|past\s+paper)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(
            r"\b(?:please|create|generate|make|build|prepare|ask|give|test|quiz|assess|challenge|mock|creat|genrate|generte|mak|preprare)\b",
            " ",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\b(?:questions?\s+should\s+be\s+there|should\s+be\s+there|there\s+should\s+be)\b", " ", clean, flags=re.IGNORECASE)

        subject_normalized = self._normalize(subject)
        if "computer science" in subject_normalized:
            clean = re.sub(r"\b(?:cs|cse|computer\s+science|computer\s+applications?)\b", " ", clean, flags=re.IGNORECASE)
        elif subject_normalized and not subject_normalized.startswith("paper"):
            for term in self._terms_for_subject(subject):
                clean = re.sub(rf"\b{re.escape(term)}\b", " ", clean, flags=re.IGNORECASE)

        clean = re.sub(r"\b(?:questions?|question|mcqs?|mock|test|exam|practice|style|styles?)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"^\s*(?:for|on|about|in|of|me|my|the|a|an)\s+", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+(?:for|on|about|in|of|me|my|the|a|an)\s*$", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"[-:]+", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip(" .,")
        clean_normalized = self._normalize(clean)
        if not clean_normalized or clean_normalized in {"paper", "paper ii", "paper 2", "paper i", "paper 1"}:
            return ""
        return self._canonical_focus_label(clean)

    def _terms_for_subject(self, subject: str) -> list[str]:
        normalized = self._normalize(subject)
        terms = [term for term in normalized.split() if len(term) >= 3]
        if normalized == "political science":
            terms.append("polity")
        return terms

    def _canonical_focus_label(self, clean: str) -> str:
        normalized = self._normalize(clean)
        if re.search(r"\b(?:dsa|data\s+structures?|data\s+structure\s+and\s+algorithms?)\b", normalized):
            suffixes: list[str] = []
            if re.search(r"\barrays?\b", normalized):
                suffixes.append("Arrays")
            if re.search(r"\bstacks?\b", normalized):
                suffixes.append("Stacks")
            if re.search(r"\bqueues?\b", normalized):
                suffixes.append("Queues")
            if re.search(r"\btrees?\b", normalized):
                suffixes.append("Trees")
            if re.search(r"\bgraphs?\b", normalized):
                suffixes.append("Graphs")
            if re.search(r"\bhash(?:ing)?\b", normalized):
                suffixes.append("Hashing")
            return "Data Structures and Algorithms" + (f" - {' / '.join(suffixes)}" if suffixes else "")
        if re.search(r"\barrays?\b", normalized):
            return "Data Structures and Algorithms - Arrays"
        if re.search(r"\bstacks?\b", normalized):
            return "Data Structures and Algorithms - Stacks"
        if re.search(r"\bqueues?\b", normalized):
            return "Data Structures and Algorithms - Queues"
        if re.search(r"\btrees?\b", normalized):
            return "Data Structures and Algorithms - Trees"
        if re.search(r"\bgraphs?\b", normalized):
            return "Data Structures and Algorithms - Graphs"
        replacements = {
            "dbms": "DBMS",
            "os": "Operating System",
            "toc": "Theory of Computation",
            "cn": "Computer Networks",
            "ai": "Artificial Intelligence",
        }
        words = []
        for word in clean.split():
            key = self._normalize(word)
            words.append(replacements.get(key, word.upper() if key in {"sql", "dbms"} else word))
        return self._dedupe_words(" ".join(words)).strip(" .")[:120]

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
            r"^\s*(please\s+)?(ask|give)\s+(me\s+)?(some\s+|a\s+)?((?:numerical|numeric|calculation[- ]?based|practice|mixed|easy|medium|hard)\s+)?(questions?|mcqs?|quiz|test)\s*(on|about|in|for)?\s*",
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
        clean = re.sub(r"\b(?:real\s+)?(?:pyq|previous\s+year|past\s+paper)\s*[- ]?styles?\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(only|based\s+on)?\s*(pyq|previous\s+year|past\s+paper|official)\s+(questions?|papers?|styles?)?\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\b(questions?\s+should\s+be\s+there|should\s+be\s+there|there\s+should\s+be)\b", " ", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s+", " ", clean)
        return self._clean_mock_topic(clean.strip(" .,"))

    def _existing_mock_topic(self, text: str) -> str:
        clean = re.sub(r"\b(mock\s+test|mock|test|where|show|open|view|see|find|latest|not\s+created|till\s+now|yet|ready|is|my|the)\b", " ", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", clean).strip(" .")

    def _source_query(self, topic: str, requires_sources: bool) -> str:
        if not requires_sources:
            return topic
        return f"{topic} previous year questions PYQ MCQ"

    def _clean_source_query(self, raw: str, topic: str, requires_sources: bool) -> str:
        if not requires_sources:
            return re.sub(r"\s+", " ", topic).strip(" .")[:240]
        clean = self._clean_mock_topic(raw)
        if not clean or len(clean.split()) > max(8, len(topic.split()) + 4):
            clean = self._source_query(topic, requires_sources)
        if requires_sources and not re.search(r"\b(pyq|previous\s+year|past\s+paper|official)\b", clean, flags=re.IGNORECASE):
            clean = f"{clean} previous year questions PYQ MCQ"
        return re.sub(r"\s+", " ", clean).strip(" .")[:240]

    def _mentions_ugc_net(self, normalized: str) -> bool:
        return bool(("ugc" in normalized and "net" in normalized) or "net ugc" in normalized)

    def _mentions_pyq(self, normalized: str) -> bool:
        return bool(re.search(r"\b(pyqs?|previous\s+year|past\s+paper|past\s+papers|official\s+questions?|year\s+questions?)\b", normalized))

    def _mentions_mock_test(self, normalized: str) -> bool:
        return bool(re.search(r"\b(mock|mocktest|practice\s+test|mock\s+exam|quiz|test|questions?|mcqs?)\b", normalized))

    def _mentions_pyq_style(self, normalized: str, payload: dict[str, Any] | None = None) -> bool:
        combined = normalized
        if payload:
            constraints = payload.get("constraints") if isinstance(payload.get("constraints"), list) else []
            combined = f"{combined} {' '.join(str(item) for item in constraints)} {payload.get('source_requirement', '')}"
        return bool(re.search(r"\b(pyq\s*(?:styles?|stiles?)|pyq-(?:styles?|stiles?)|previous\s+year\s+(?:styles?|stiles?)|past\s+paper\s+(?:styles?|stiles?))\b", combined, flags=re.IGNORECASE))

    def _is_mock_generation_intent(self, normalized: str) -> bool:
        if re.search(r"\b(not created|not generated|did not create|didnt create|not showing|till now|where|show|open|view|see|find)\b", normalized):
            return False
        if self._is_assessment_generation_intent(normalized):
            return True
        has_action = bool(re.search(r"\b(create|generate|make|build|prepare|new|another|creat|genrate|generte|mak|preprare)\b", normalized))
        has_object = bool(re.search(r"\b(mock|exam|practice|test|quiz|questions?|mcqs?|pyqs?)\b", normalized))
        return has_action and has_object and not re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized)

    def _is_assessment_generation_intent(self, normalized: str) -> bool:
        if re.search(r"\b(frontend|backend|pytest|lint|unit|integration)\b", normalized):
            return False
        if re.search(r"\b(test|quiz|assess|challenge)\s+(me|my knowledge)?\b", normalized):
            return True
        if re.search(r"\b(ask|give)\s+(me\s+)?(some\s+|a\s+)?((?:numerical|numeric|calculation[- ]?based|practice|mixed|easy|medium|hard)\s+)?(questions?|mcqs?|quiz)\b", normalized):
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
        match = re.search(r"\b(\d{1,2})\s+(?:easy|medium|hard|mixed|tough|advanced|basic)\s+(?:questions?|q|mcqs?)\b", normalized)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d{1,2})\s*[- ]?(?:questions?|q|mcqs?)\b", normalized)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(?:questions?|mcqs?)\s*[:=-]?\s*(\d{1,2})\b", normalized)
        if match:
            return int(match.group(1))
        words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "fifteen": 15,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
        }
        match = re.search(r"\b(" + "|".join(words) + r")\s+(?:questions?|mcqs?)\b", normalized)
        return words[match.group(1)] if match else None

    def _duration_from_text(self, normalized: str) -> int | None:
        match = re.search(r"\b(\d{1,3})\s*(?:minutes?|mins?|min)\b", normalized)
        if match:
            return int(match.group(1))
        match = re.search(r"\b(\d{1,2})\s*(?:hours?|hrs?|hr)\b", normalized)
        return int(match.group(1)) * 60 if match else None

    def _difficulty_from_text(self, normalized: str) -> str | None:
        match = re.search(r"\b(easy|medium|hard|mixed)\s+(?:difficulty|level)\b", normalized)
        if match:
            return match.group(1)
        match = re.search(r"\b(?:difficulty|level)\s*[:=-]?\s*(easy|medium|hard|mixed)\b", normalized)
        if match:
            return match.group(1)
        match = re.search(r"\b(easy|medium|hard|mixed)\s+(?:questions?|mcqs?|mock\s+test|test|quiz)\b", normalized)
        if match:
            return match.group(1)
        match = re.search(r"\b(?:questions?|mcqs?)\s+(easy|medium|hard|mixed)\b", normalized)
        if match:
            return match.group(1)
        match = re.search(r"\b(easy|medium|hard|mixed)\s+\d{1,2}\s*(?:questions?|q|mcqs?)\b", normalized)
        if match:
            return match.group(1)
        match = re.search(r"\b(easy|medium|hard|mixed)\b\s*$", normalized)
        if match:
            return match.group(1)
        if re.search(r"\b(tough|toughest|advanced|challenging)\b", normalized):
            return "hard"
        if re.search(r"\b(beginner|basic|simple)\b", normalized):
            return "easy"
        return None

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
