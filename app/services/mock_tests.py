import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import (
    MockAttempt,
    MockQuestion,
    MockQuestionReview,
    MockQuestionView,
    MockTest,
    MockTestDifficulty,
    MockTestGenerateRequest,
    MockTestMode,
    MockTestSubmitRequest,
    MockTestSubmitResponse,
    MockTestView,
    Source,
)
from app.services.documents import DocumentService
from app.services.llm import LlmService
from app.services.mock_intelligence import MockBlueprint, MockTestIntelligenceService
from app.services.search import SearchService


MOCK_TEST_GENERATION_TIMEOUT_SECONDS = 75


class MockTestSourceMaterialError(ValueError):
    def __init__(self, message: str, source_actions: list[str]):
        super().__init__(message)
        self.source_actions = source_actions


class MockTestService:
    def __init__(self, settings: Settings, llm: LlmService, documents: DocumentService | None = None, search: SearchService | None = None):
        base = Path(settings.data_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.mock_tests_dir = base / "mock-tests"
        self.tests_dir = self.mock_tests_dir / "tests"
        self.attempts_dir = self.mock_tests_dir / "attempts"
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.llm = llm
        self.documents = documents
        self.search = search
        self.intelligence = MockTestIntelligenceService(settings, llm, documents, search)

    def list_tests(self) -> list[MockTestView]:
        tests: list[MockTestView] = []
        for path in self.tests_dir.glob("*.json"):
            test = self._read_test_path(path)
            if test:
                tests.append(self._view(test))
        return sorted(tests, key=lambda item: item.created_at, reverse=True)

    def get_test(self, test_id: str) -> MockTestView | None:
        test = self._read_test(test_id)
        return self._view(test) if test else None

    async def generate(self, request: MockTestGenerateRequest) -> tuple[MockTestView, list[str]]:
        clean = self._clean_request(request)
        llm_configured = self.llm.model_configured()
        setup_required: list[str] = []
        questions: list[MockQuestion] = []
        source = "llm"
        source_refs = list(clean.source_refs)
        source_material = clean.source_text.strip()
        quality_score = 0.0
        quality_warnings: list[str] = []
        blueprint = MockBlueprint(
            exam=clean.exam,
            subject=clean.subject,
            topic=clean.topic,
            generation_mode="topic_practice",
            blueprint_source="request",
            syllabus_units=[clean.topic],
            expected_terms=self._terms(f"{clean.topic} {clean.source_query}"),
        )

        if clean.source_requirement != "none":
            collected_text, collected_sources, source_setup = await self._collect_source_material(clean)
            setup_required.extend(source_setup)
            source_refs = self._dedupe_sources([*source_refs, *collected_sources])
            source_material = "\n\n".join(part for part in [source_material, collected_text] if part).strip()
            source = "pyq_required" if clean.source_requirement == "pyq_required" else "source_backed"
            blueprint.generation_mode = "source_backed_pyq" if clean.source_requirement == "pyq_required" else "syllabus_based"
            blueprint.blueprint_source = "source_material"
            blueprint.sources = source_refs[:8]
            blueprint.source_text = source_material[:5000]
            if not source_material:
                label = "PYQ" if clean.source_requirement == "pyq_required" else "source-backed"
                raise MockTestSourceMaterialError(
                    f"{label} mock tests need actual question source material first. Upload a relevant PYQ PDF or switch to PYQ-style practice.",
                    self._source_failure_actions(clean),
                )
            if not self._has_usable_question_material(source_material, clean.question_count):
                raise MockTestSourceMaterialError(
                    "I found links or snippets, but not enough actual PYQ question text to make a verified test. Upload a clearer PYQ PDF or switch to PYQ-style practice.",
                    self._source_failure_actions(clean),
                )
            if not llm_configured:
                missing_setup = self.llm.missing_setup_for_model()
                raise ValueError(f"Source-backed mock tests found source material, but {missing_setup} is required to convert it into verified questions.")
            raw, llm_setup = await self.llm.complete(self._source_system_prompt(clean), self._source_user_prompt(clean, source_material, source_refs))
            setup_required.extend(llm_setup)
            questions = self._questions_from_llm(raw, clean, default_sources=source_refs[:3])
            questions, quality_score, quality_warnings = self.intelligence.validator.validate(
                questions,
                blueprint,
                clean.question_count,
                strict_terms=False,
                requested_difficulty=clean.difficulty,
                constraints=clean.constraints,
            )
            if len(questions) < clean.question_count:
                raise MockTestSourceMaterialError(
                    "The source looked relevant, but Astra could not extract enough usable MCQs without inventing questions. Upload a clearer PYQ PDF or switch to PYQ-style practice.",
                    self._source_failure_actions(clean),
                )
        else:
            blueprint, blueprint_setup = await self.intelligence.build_blueprint(clean)
            setup_required.extend(blueprint_setup)
            source_refs = self._dedupe_sources([*source_refs, *blueprint.sources])
            source = blueprint.generation_mode

            if llm_configured:
                candidate_questions: list[MockQuestion] = []
                best_accepted: list[MockQuestion] = []
                rejected_reasons: list[str] = []
                best_score = 0.0
                for attempt in range(self._generation_attempt_count(clean.question_count)):
                    remaining_count = max(1, clean.question_count - len(best_accepted))
                    candidate_count = self._candidate_count(clean.question_count, attempt, remaining_count)
                    raw, llm_setup = await self.llm.complete(
                        self.intelligence.generation_system_prompt(),
                        self.intelligence.generation_user_prompt(
                            clean,
                            blueprint,
                            candidate_count=candidate_count,
                            rejected_reasons=rejected_reasons,
                            accepted_count=len(best_accepted),
                        ),
                    )
                    setup_required.extend(llm_setup)
                    if any(item in {"CEREBRAS_API", "CEREBRAS_SDK", "NVIDIA_API"} for item in llm_setup):
                        rejected_reasons = [f"LLM provider failed: {', '.join(llm_setup)}"]
                        break
                    new_questions = self._questions_from_llm(raw, clean, max_questions=candidate_count)
                    candidate_questions = self._dedupe_questions([*candidate_questions, *new_questions])
                    accepted, score, warnings = self.intelligence.validator.validate(
                        candidate_questions,
                        blueprint,
                        clean.question_count,
                        strict_terms=bool(blueprint.expected_terms),
                        requested_difficulty=clean.difficulty,
                        constraints=clean.constraints,
                    )
                    if len(accepted) > len(best_accepted):
                        best_accepted = accepted
                    best_score = max(best_score, score)
                    rejected_reasons = warnings
                    if len(accepted) >= clean.question_count:
                        questions = self._renumber_questions(accepted)
                        quality_score = score
                        quality_warnings = warnings
                        source = "llm_planned" if blueprint.generation_mode == "llm_planned" else blueprint.generation_mode
                        break
                    quality_score = max(quality_score, best_score)
                    quality_warnings = warnings

                if len(questions) < clean.question_count:
                    best_accepted, best_score, rejected_reasons, rescue_setup = await self._single_question_rescue(
                        clean,
                        blueprint,
                        candidate_questions,
                        best_accepted,
                        rejected_reasons,
                    )
                    setup_required.extend(rescue_setup)
                    if len(best_accepted) >= clean.question_count:
                        questions = self._renumber_questions(best_accepted)
                        quality_score = best_score
                        quality_warnings = rejected_reasons
                        source = "llm_planned" if blueprint.generation_mode == "llm_planned" else blueprint.generation_mode
                    else:
                        quality_score = max(quality_score, best_score)
                        quality_warnings = rejected_reasons

            if len(questions) < clean.question_count:
                candidate_questions = self.intelligence.profile_questions(clean, blueprint)
                accepted, score, warnings = self.intelligence.validator.validate(
                    candidate_questions,
                    blueprint,
                    clean.question_count,
                    strict_terms=bool(blueprint.expected_terms),
                    requested_difficulty=clean.difficulty,
                    constraints=clean.constraints,
                )
                if len(accepted) >= clean.question_count:
                    questions = self._renumber_questions(accepted)
                    quality_score = score
                    quality_warnings = warnings
                    source = blueprint.generation_mode

            if len(questions) < clean.question_count:
                warning_text = " ".join(quality_warnings[:2])
                detail = f" {warning_text}" if warning_text else ""
                setup = sorted(set(setup_required or ([] if llm_configured else [self.llm.missing_setup_for_model()])))
                setup_label = f" Missing setup: {', '.join(setup)}." if setup else ""
                raise ValueError(
                    f"Astra could not create enough exam-quality MCQs for {clean.topic} without falling back to generic filler.{detail}{setup_label}"
                )

        if len(questions) >= clean.question_count:
            setup_required = [
                item
                for item in setup_required
                if item not in {"CEREBRAS_API", "CEREBRAS_SDK", "CEREBRAS_API_KEY", "NVIDIA_API", "NVIDIA_API_KEY", "TAVILY_API_KEY"}
            ]

        test = MockTest(
            id=uuid.uuid4().hex,
            topic=clean.topic,
            exam=blueprint.exam or clean.exam,
            subject=blueprint.subject or clean.subject,
            mode=clean.mode,
            difficulty=clean.difficulty,
            question_count=clean.question_count,
            duration_minutes=clean.duration_minutes,
            questions=self._renumber_questions(questions[: clean.question_count]),
            created_at=datetime.utcnow(),
            source=source,
            generation_mode=blueprint.generation_mode,
            blueprint_source=blueprint.blueprint_source,
            syllabus_units=blueprint.syllabus_units[:16],
            quality_score=quality_score,
            quality_warnings=quality_warnings[:12],
            source_requirement=clean.source_requirement,
            source_mode=clean.source_mode,
            source_query=clean.source_query,
            constraints=clean.constraints,
            sources=source_refs[:8],
            setup_required=setup_required,
        )
        self._write_test(test)
        return self._view(test), setup_required

    def start_attempt(self, test_id: str) -> tuple[MockTestView, MockAttempt]:
        test = self._read_test(test_id)
        if not test:
            raise ValueError("Mock test not found.")
        attempt = MockAttempt(id=uuid.uuid4().hex, test_id=test.id, status="active", started_at=datetime.utcnow())
        self._write_attempt(attempt)
        return self._view(test), attempt

    def submit_attempt(self, test_id: str, attempt_id: str, request: MockTestSubmitRequest) -> MockTestSubmitResponse:
        test = self._read_test(test_id)
        if not test:
            raise ValueError("Mock test not found.")
        attempt = self._read_attempt(attempt_id)
        if not attempt or attempt.test_id != test.id:
            raise ValueError("Mock attempt not found.")

        answers: dict[str, int] = {}
        for question in test.questions:
            value = request.answers.get(question.id)
            if isinstance(value, int) and 0 <= value < len(question.options):
                answers[question.id] = value

        review: list[MockQuestionReview] = []
        score = 0
        for question in test.questions:
            selected = answers.get(question.id)
            is_correct = selected == question.correct_option_index
            if is_correct:
                score += 1
            review.append(
                MockQuestionReview(
                    **self._question_view(question).model_dump(),
                    correct_option_index=question.correct_option_index,
                    selected_option_index=selected,
                    is_correct=is_correct,
                    explanation=question.explanation,
                )
            )

        elapsed = max(0, int(request.elapsed_seconds))
        attempt.status = "submitted"
        attempt.submitted_at = datetime.utcnow()
        attempt.elapsed_seconds = elapsed
        attempt.answers = answers
        self._write_attempt(attempt)

        total = len(test.questions)
        percentage = round((score / total) * 100, 2) if total else 0.0
        return MockTestSubmitResponse(
            test=self._view(test),
            attempt=attempt,
            score=score,
            total=total,
            percentage=percentage,
            correct_count=score,
            incorrect_count=max(0, total - score),
            elapsed_seconds=elapsed,
            review=review,
        )

    def _clean_request(self, request: MockTestGenerateRequest) -> MockTestGenerateRequest:
        topic = re.sub(r"\s+", " ", request.topic).strip(" .") or "general aptitude"
        exam = re.sub(r"\s+", " ", request.exam).strip(" .")[:80]
        subject = re.sub(r"\s+", " ", request.subject).strip(" .")[:80]
        source_requirement = request.source_requirement if request.source_requirement in {"none", "pyq_required", "source_backed"} else "none"
        source_mode = "uploaded_docs"
        constraints = [str(item).strip()[:80] for item in request.constraints if str(item).strip()]
        source_query = re.sub(r"\s+", " ", request.source_query).strip(" .") or (f"{topic} previous year questions PYQ MCQ" if source_requirement != "none" else topic)
        return MockTestGenerateRequest(
            topic=topic[:120],
            exam=exam,
            subject=subject,
            question_count=max(1, min(50, request.question_count or 10)),
            difficulty=request.difficulty if request.difficulty in {"easy", "medium", "hard", "mixed"} else "mixed",
            mode="mcq",
            duration_minutes=max(1, min(180, request.duration_minutes or 20)),
            source_requirement=source_requirement,
            source_mode=source_mode,
            source_query=source_query[:240],
            constraints=constraints[:8],
            source_text=request.source_text[:12000],
            source_refs=request.source_refs[:8],
        )

    def _system_prompt(self) -> str:
        return (
            "You are Astra's mock test generator. Return strict JSON only. "
            "Create exam-ready MCQ questions with exactly four options and one correct answer. "
            "Do not include Markdown or prose outside JSON."
        )

    def _user_prompt(self, request: MockTestGenerateRequest) -> str:
        return json.dumps(
            {
                "topic": request.topic,
                "question_count": request.question_count,
                "difficulty": request.difficulty,
                "mode": request.mode,
                "schema": {
                    "questions": [
                        {
                            "prompt": "question text",
                            "options": ["A", "B", "C", "D"],
                            "correct_option_index": 0,
                            "explanation": "short explanation",
                            "difficulty": "easy|medium|hard",
                            "tags": ["tag"],
                        }
                    ]
                },
            },
            ensure_ascii=True,
        )

    def _source_system_prompt(self, request: MockTestGenerateRequest) -> str:
        source_label = "previous-year/PYQ" if request.source_requirement == "pyq_required" else "source-backed"
        return (
            "You are Astra's source-grounded mock test generator. Return strict JSON only. "
            f"Create {source_label} MCQs using only the provided source material. "
            "Do not invent questions or facts that are not supported by the source text. "
            "Do not create generic study-advice questions. "
            "If the source material does not contain enough usable question/evidence material, return {\"questions\": []}."
        )

    def _source_user_prompt(self, request: MockTestGenerateRequest, source_material: str, sources: list[Source]) -> str:
        return json.dumps(
            {
                "topic": request.topic,
                "question_count": request.question_count,
                "difficulty": request.difficulty,
                "difficulty_policy": self.intelligence._difficulty_policy(request.difficulty),
                "difficulty_sequence": self.intelligence._difficulty_sequence(request.difficulty, request.question_count),
                "mode": request.mode,
                "constraints": request.constraints,
                "source_requirement": request.source_requirement,
                "source_query": request.source_query,
                "sources": [source.model_dump(mode="json") for source in sources],
                "source_material": source_material[:12000],
                "schema": {
                    "questions": [
                        {
                            "prompt": "question text grounded in source",
                            "options": ["A", "B", "C", "D"],
                            "correct_option_index": 0,
                            "explanation": "short source-backed explanation",
                            "difficulty": "easy|medium|hard",
                            "tags": ["tag"],
                        }
                    ]
                },
            },
            ensure_ascii=True,
        )

    def _questions_from_llm(
        self,
        raw: str | None,
        request: MockTestGenerateRequest,
        default_sources: list[Source] | None = None,
        max_questions: int | None = None,
    ) -> list[MockQuestion]:
        payload = self._parse_json(raw)
        raw_questions = payload.get("questions") if payload else None
        if not isinstance(raw_questions, list):
            raw_questions = self._partial_question_items(raw)

        questions: list[MockQuestion] = []
        limit = max(1, min(80, max_questions or request.question_count))
        for index, item in enumerate(raw_questions[:limit]):
            if not isinstance(item, dict):
                continue
            question = self._coerce_question(item, request, index, default_sources or [])
            if question:
                questions.append(question)
        return questions

    def _partial_question_items(self, raw: str | None) -> list[dict[str, Any]]:
        if not isinstance(raw, str) or not raw.strip():
            return []
        questions_key = re.search(r'"questions"\s*:', raw)
        if not questions_key:
            return []
        array_start = raw.find("[", questions_key.end())
        if array_start < 0:
            return []

        items: list[dict[str, Any]] = []
        in_string = False
        escaped = False
        depth = 0
        start: int | None = None
        for index, char in enumerate(raw[array_start + 1 :], start=array_start + 1):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
                continue
            if char == "}":
                if depth <= 0:
                    continue
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        item = json.loads(raw[start : index + 1])
                    except Exception:
                        item = None
                    if isinstance(item, dict):
                        items.append(item)
                    start = None
                continue
            if char == "]" and depth == 0:
                break
        return items

    def _generation_attempt_count(self, requested_count: int) -> int:
        return min(8, max(5, (requested_count + 7) // 8 + 3))

    def _candidate_count(self, requested_count: int, attempt: int, remaining_count: int | None = None) -> int:
        remaining = max(1, remaining_count or requested_count)
        extra = 4 if attempt == 0 else 6
        return min(16, max(4, remaining + extra))

    async def _single_question_rescue(
        self,
        request: MockTestGenerateRequest,
        blueprint: MockBlueprint,
        candidate_questions: list[MockQuestion],
        best_accepted: list[MockQuestion],
        rejected_reasons: list[str],
    ) -> tuple[list[MockQuestion], float, list[str], list[str]]:
        setup_required: list[str] = []
        all_candidates = self._dedupe_questions(candidate_questions)
        best = list(best_accepted)
        best_score = round(len(best) / max(1, request.question_count), 2)
        warnings = list(rejected_reasons)
        max_calls = min(24, max(6, (request.question_count - len(best)) * 3 + 4))
        for _ in range(max_calls):
            if len(best) >= request.question_count:
                break
            raw, llm_setup = await self.llm.complete(
                self.intelligence.generation_system_prompt(),
                self.intelligence.generation_user_prompt(
                    request,
                    blueprint,
                    candidate_count=1,
                    rejected_reasons=warnings,
                    accepted_count=len(best),
                ),
            )
            setup_required.extend(llm_setup)
            if any(item in {"CEREBRAS_API", "CEREBRAS_SDK", "NVIDIA_API"} for item in llm_setup):
                warnings = [f"LLM provider failed: {', '.join(llm_setup)}"]
                break
            new_questions = self._questions_from_llm(raw, request, max_questions=1)
            if new_questions:
                all_candidates = self._dedupe_questions([*all_candidates, *new_questions])
            accepted, score, current_warnings = self.intelligence.validator.validate(
                all_candidates,
                blueprint,
                request.question_count,
                strict_terms=bool(blueprint.expected_terms),
                requested_difficulty=request.difficulty,
                constraints=request.constraints,
            )
            if len(accepted) > len(best):
                best = accepted
            best_score = max(best_score, score)
            warnings = current_warnings or warnings
        return best, best_score, warnings, sorted(set(setup_required))

    def _dedupe_questions(self, questions: list[MockQuestion]) -> list[MockQuestion]:
        seen: set[str] = set()
        unique: list[MockQuestion] = []
        for question in questions:
            key = re.sub(r"[^a-z0-9]+", " ", question.prompt.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(question)
        return self._renumber_questions(unique)

    def _renumber_questions(self, questions: list[MockQuestion]) -> list[MockQuestion]:
        return [question.model_copy(update={"id": f"q{index + 1}"}) for index, question in enumerate(questions)]

    def _coerce_question(self, item: dict[str, Any], request: MockTestGenerateRequest, index: int, default_sources: list[Source] | None = None) -> MockQuestion | None:
        prompt = str(item.get("prompt") or item.get("question") or "").strip()
        options = item.get("options")
        if not prompt or not isinstance(options, list):
            return None
        clean_options = [str(option).strip() for option in options if str(option).strip()][:4]
        if len(clean_options) != 4:
            return None
        try:
            correct_index = int(item.get("correct_option_index", item.get("answer_index", 0)))
        except (TypeError, ValueError):
            return None
        if not 0 <= correct_index <= 3:
            return None
        difficulty = str(item.get("difficulty") or "").strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium" if request.difficulty == "mixed" else request.difficulty
        tags_raw = item.get("tags")
        tags = [str(tag).strip()[:32] for tag in tags_raw if str(tag).strip()] if isinstance(tags_raw, list) else []
        explanation = str(item.get("explanation") or f"Option {chr(65 + correct_index)} is the best answer.").strip()
        source_refs = self._question_sources(item, default_sources or [])
        return MockQuestion(
            id=f"q{index + 1}",
            prompt=prompt[:600],
            options=clean_options,
            correct_option_index=correct_index,
            explanation=explanation[:900],
            difficulty=difficulty,  # type: ignore[arg-type]
            tags=tags[:4] or [request.topic[:32]],
            source_refs=source_refs,
        )

    async def _collect_source_material(self, request: MockTestGenerateRequest) -> tuple[str, list[Source], list[str]]:
        setup_required: list[str] = []
        chunks: list[str] = []
        sources: list[Source] = []

        document_text, document_sources = self._collect_document_sources(request)
        if document_text:
            chunks.append(document_text)
            sources.extend(document_sources)

        return "\n\n".join(chunks).strip(), self._dedupe_sources(sources), sorted(set(setup_required))

    def _collect_document_sources(self, request: MockTestGenerateRequest) -> tuple[str, list[Source]]:
        if not self.documents:
            return "", []
        terms = {term for term in re.findall(r"[a-zA-Z0-9]{3,}", f"{request.topic} {request.source_query}".lower())}
        scored: list[tuple[int, Any, str]] = []
        for record in self.documents.list_documents():
            excerpt = self.documents.document_excerpt(record.id, max_chars=5000)
            if not excerpt.strip():
                continue
            lowered = f"{record.title} {excerpt}".lower()
            score = sum(lowered.count(term) for term in terms)
            scored.append((score, record, excerpt))

        selected = [item for item in sorted(scored, key=lambda item: item[0], reverse=True) if item[0] > 0][:3]
        if not selected and scored:
            selected = sorted(scored, key=lambda item: item[0], reverse=True)[:1]

        sources: list[Source] = []
        chunks: list[str] = []
        for _, record, excerpt in selected:
            sources.append(Source(title=record.title, url=f"document:{record.id}", snippet=record.text_preview, provider="Uploaded PDF"))
            chunks.append(f"{record.title} (Uploaded PDF):\n{excerpt[:5000]}")
        return "\n\n".join(chunks).strip(), sources

    def _question_sources(self, item: dict[str, Any], default_sources: list[Source]) -> list[Source]:
        raw_sources = item.get("source_refs")
        sources: list[Source] = []
        if isinstance(raw_sources, list):
            for raw in raw_sources:
                if isinstance(raw, dict):
                    try:
                        sources.append(Source.model_validate(raw))
                    except Exception:
                        continue
        return self._dedupe_sources([*sources, *default_sources])[:3]

    def _has_usable_question_material(self, source_material: str, question_count: int) -> bool:
        text = re.sub(r"\s+", " ", source_material)
        question_like = len(
            re.findall(
                r"\b(?:which|what|who|where|when|why|how|consider|choose|select|find|identify|calculate|determine)\b.{8,220}\?",
                text,
                flags=re.IGNORECASE,
            )
        )
        numbered_questions = len(re.findall(r"\b(?:q(?:uestion)?\.?\s*)?\d{1,3}\s*[\).:-]\s+.{12,220}\?", text, flags=re.IGNORECASE))
        option_markers = len(re.findall(r"(?:\b[A-D]\s*[\).:-]|\([A-D]\))\s*.{2,120}", text, flags=re.IGNORECASE))
        answer_markers = len(re.findall(r"\b(?:answer|ans\.?|correct option)\b", text, flags=re.IGNORECASE))
        pyq_markers = len(re.findall(r"\b(?:pyq|previous year|past paper|ugc net|net ugc)\b", text, flags=re.IGNORECASE))
        score = question_like + numbered_questions + min(option_markers // 4, question_count) + min(answer_markers, question_count) + min(pyq_markers, 2)
        needed = max(3, min(8, question_count // 2))
        return score >= needed and (question_like + numbered_questions) >= 2

    def _source_failure_actions(self, request: MockTestGenerateRequest) -> list[str]:
        return ["Use uploaded PDF", "Generate PYQ-style practice"]

    def _terms(self, text: str) -> list[str]:
        return [term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]{2,}", text.lower())][:60]

    def _dedupe_sources(self, sources: list[Source]) -> list[Source]:
        seen: set[str] = set()
        unique: list[Source] = []
        for source in sources:
            key = (source.url or source.title).lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(source)
        return unique

    def _view(self, test: MockTest) -> MockTestView:
        return MockTestView(
            id=test.id,
            topic=test.topic,
            exam=test.exam,
            subject=test.subject,
            mode=test.mode,
            difficulty=test.difficulty,
            question_count=len(test.questions),
            duration_minutes=test.duration_minutes,
            questions=[self._question_view(question) for question in test.questions],
            created_at=test.created_at,
            source=test.source,
            generation_mode=test.generation_mode,
            blueprint_source=test.blueprint_source,
            syllabus_units=test.syllabus_units,
            quality_score=test.quality_score,
            quality_warnings=test.quality_warnings,
            source_requirement=test.source_requirement,
            source_mode=test.source_mode,
            source_query=test.source_query,
            constraints=test.constraints,
            sources=test.sources,
            setup_required=test.setup_required,
        )

    def _question_view(self, question: MockQuestion) -> MockQuestionView:
        return MockQuestionView(
            id=question.id,
            prompt=question.prompt,
            options=question.options,
            difficulty=question.difficulty,
            tags=question.tags,
            source_refs=question.source_refs,
        )

    def _parse_json(self, raw: str | None) -> dict[str, Any] | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
        decoder = json.JSONDecoder()
        candidates = [raw.strip()]
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        if fence:
            candidates.append(fence.group(1).strip())
        brace_index = raw.find("{")
        if brace_index >= 0:
            candidates.append(raw[brace_index:].strip())
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                try:
                    payload, _ = decoder.raw_decode(candidate)
                except Exception:
                    continue
            if isinstance(payload, dict):
                return payload
        return None

    def _test_path(self, test_id: str) -> Path:
        return self.tests_dir / f"{test_id}.json"

    def _attempt_path(self, attempt_id: str) -> Path:
        return self.attempts_dir / f"{attempt_id}.json"

    def _write_test(self, test: MockTest) -> None:
        self._test_path(test.id).write_text(json.dumps(test.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _read_test(self, test_id: str) -> MockTest | None:
        return self._read_test_path(self._test_path(test_id))

    def _read_test_path(self, path: Path) -> MockTest | None:
        if not path.exists():
            return None
        try:
            return MockTest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def _write_attempt(self, attempt: MockAttempt) -> None:
        self._attempt_path(attempt.id).write_text(json.dumps(attempt.model_dump(mode="json"), indent=2), encoding="utf-8")

    def _read_attempt(self, attempt_id: str) -> MockAttempt | None:
        path = self._attempt_path(attempt_id)
        if not path.exists():
            return None
        try:
            return MockAttempt.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None
