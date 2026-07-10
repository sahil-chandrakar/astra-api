from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.models import MockQuestion, MockTestDifficulty, MockTestGenerateRequest, Source
from app.services.documents import DocumentService
from app.services.llm import LlmService
from app.services.search import SearchService


@dataclass(frozen=True)
class ExamProfile:
    id: str
    exam: str
    subject: str
    aliases: tuple[str, ...]
    syllabus_units: tuple[str, ...]
    question_style: str
    expected_terms: tuple[str, ...]


@dataclass
class MockBlueprint:
    exam: str = ""
    subject: str = ""
    topic: str = ""
    generation_mode: str = "topic_practice"
    blueprint_source: str = "local_profile"
    syllabus_units: list[str] = field(default_factory=list)
    question_style: str = "Conceptual MCQ with one clearly correct answer."
    expected_terms: list[str] = field(default_factory=list)
    confidence: float = 0.75
    sources: list[Source] = field(default_factory=list)
    source_text: str = ""
    quality_warnings: list[str] = field(default_factory=list)


class MockQuestionQualityValidator:
    weak_prompt_patterns = (
        r"\bcore purpose of\b",
        r"\bprioritize first\b",
        r"\bpractice style\b",
        r"\bgood answer\b",
        r"\bhabit improves performance\b",
        r"\blooks unfamiliar\b",
        r"\bwhy are explanations important\b",
        r"\bgetting .* wrong\b",
        r"\bmetric matters most\b",
        r"\bhow should difficulty\b",
    )
    weak_option_patterns = (
        r"\brandom guessing\b",
        r"\bskipping examples\b",
        r"\bunrelated theory\b",
        r"\bonly memorizing\b",
        r"\bavoiding practice\b",
        r"\bclear fundamentals\b",
        r"\bwatching summaries\b",
        r"\bignore the mistake\b",
    )

    def validate(
        self,
        questions: list[MockQuestion],
        blueprint: MockBlueprint,
        required_count: int,
        strict_terms: bool = True,
        requested_difficulty: MockTestDifficulty = "mixed",
        constraints: list[str] | None = None,
    ) -> tuple[list[MockQuestion], float, list[str]]:
        warnings: list[str] = []
        accepted: list[MockQuestion] = []
        expected_terms = {self._normalize(term) for term in blueprint.expected_terms if len(self._normalize(term)) >= 3}
        clean_constraints = [str(item).strip().lower() for item in constraints or [] if str(item).strip()]

        for question in questions:
            reason = self._reject_reason(question, expected_terms, strict_terms, requested_difficulty, clean_constraints)
            if reason:
                warnings.append(f"{question.id}: {reason}")
                continue
            accepted.append(question)

        score = round(len(accepted) / max(1, required_count), 2)
        if len(accepted) < required_count:
            warnings.append(f"Accepted {len(accepted)} of {required_count} generated questions after quality checks.")
        return accepted[:required_count], min(1.0, score), warnings[:12]

    def _reject_reason(
        self,
        question: MockQuestion,
        expected_terms: set[str],
        strict_terms: bool,
        requested_difficulty: MockTestDifficulty,
        constraints: list[str],
    ) -> str:
        prompt = question.prompt.strip()
        combined = " ".join([question.prompt, question.explanation, *question.options, *question.tags]).lower()
        if len(prompt.split()) < 6:
            return "question is too short"
        if requested_difficulty != "mixed" and question.difficulty != requested_difficulty:
            return f"expected {requested_difficulty} difficulty"
        if any(re.search(pattern, prompt, flags=re.IGNORECASE) for pattern in self.weak_prompt_patterns):
            return "generic study-advice question"
        if any(any(re.search(pattern, option, flags=re.IGNORECASE) for pattern in self.weak_option_patterns) for option in question.options):
            return "generic or weak answer options"
        if len({option.lower() for option in question.options}) != 4:
            return "duplicate options"
        if len(question.explanation.split()) < 6:
            return "explanation is too thin"
        if strict_terms and expected_terms:
            normalized_combined = self._normalize(combined)
            if not any(term in normalized_combined for term in expected_terms):
                return "missing blueprint concept terms"
        constraint_reason = self._constraint_reject_reason(question, constraints)
        if constraint_reason:
            return constraint_reason
        difficulty_reason = self._difficulty_reject_reason(question, requested_difficulty)
        if difficulty_reason:
            return difficulty_reason
        return ""

    def _constraint_reject_reason(self, question: MockQuestion, constraints: list[str]) -> str:
        if not constraints:
            return ""
        combined_constraints = " ".join(constraints)
        if not re.search(r"\b(numerical|numeric|calculation|calculate|quantitative|problem[- ]?solving)\b", combined_constraints):
            return ""

        prompt = question.prompt.lower()
        options_text = " ".join(question.options).lower()
        explanation = question.explanation.lower()
        quantitative_prompt = bool(
            re.search(
                r"\b(calculate|find|determine|how many|number of|value of|probability|ways|minimum|maximum|degree|edges?|vertices|subsets?|arrangements?|permutations?|combinations?)\b",
                prompt,
            )
        )
        has_numbers_or_symbols = bool(re.search(r"\d|[=+*/^<>]|\bmod\b|\bp\(|\bc\(|\bn\b", f"{prompt} {options_text} {explanation}"))
        numeric_options = sum(1 for option in question.options if re.search(r"\d|[=+*/^<>]", option.lower()))
        if not quantitative_prompt or not has_numbers_or_symbols or numeric_options < 2:
            return "does not satisfy numerical-only constraint"
        return ""

    def _difficulty_reject_reason(self, question: MockQuestion, requested_difficulty: MockTestDifficulty) -> str:
        prompt = question.prompt.strip()
        word_count = len(prompt.split())
        if requested_difficulty == "easy":
            if word_count > 95 or prompt.count("\n") > 8:
                return "too complex for easy difficulty"
        if requested_difficulty == "hard":
            hard_marker = re.search(
                r"\b(consider|given|if|after|edge case|trace|output|calculate|determine|evaluate|schedule|serializable|minimum|maximum|which statement|code|scenario)\b",
                prompt,
                flags=re.IGNORECASE,
            )
            direct_recall = re.match(r"^\s*(what is|which term|who is|which body|which protocol|which keyword)\b", prompt, flags=re.IGNORECASE)
            if direct_recall and not hard_marker:
                return "too direct for hard difficulty"
            if word_count < 10 and not hard_marker:
                return "too short for hard difficulty"
        return ""

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


class MockTestIntelligenceService:
    def __init__(self, settings: Settings, llm: LlmService, documents: DocumentService | None = None, search: SearchService | None = None):
        self.settings = settings
        self.llm = llm
        self.documents = documents
        self.search = search
        self.profiles = self._profiles()
        self.validator = MockQuestionQualityValidator()

    async def build_blueprint(self, request: MockTestGenerateRequest) -> tuple[MockBlueprint, list[str]]:
        setup_required: list[str] = []
        profile = self._match_profile(request)
        if profile:
            return self._blueprint_from_profile(profile, request), setup_required

        doc_blueprint = self._blueprint_from_uploaded_syllabus(request)
        if doc_blueprint:
            return doc_blueprint, setup_required

        if self.llm.provider_for_model() == "nvidia":
            return self._blueprint_from_request(request, "request:nvidia_fast_path"), setup_required

        llm_blueprint, llm_setup = await self._blueprint_from_llm(request)
        setup_required.extend(llm_setup)
        if llm_blueprint:
            return llm_blueprint, []

        return self._blueprint_from_request(request, "missing_llm"), sorted(set(setup_required or [self.llm.missing_setup_for_model()]))

    def _blueprint_from_request(self, request: MockTestGenerateRequest, source: str) -> MockBlueprint:
        topic = request.topic.strip() or "General Practice"
        terms = self._terms(" ".join([topic, request.source_query, *request.constraints]))
        return MockBlueprint(
            topic=topic,
            generation_mode="topic_practice",
            blueprint_source=source,
            syllabus_units=[topic],
            expected_terms=terms,
            confidence=0.35 if source != "missing_llm" else 0.2,
            quality_warnings=["Using request terms as the generation blueprint."],
        )

    def profile_questions(self, request: MockTestGenerateRequest, blueprint: MockBlueprint) -> list[MockQuestion]:
        pool = self._question_pool()
        selected_templates: list[dict[str, Any]] = []
        unit_text = " ".join([request.topic, blueprint.exam, blueprint.subject, *blueprint.syllabus_units]).lower()
        focused_keys = self._focused_template_keys(request.topic, pool)
        if focused_keys:
            for key in focused_keys:
                selected_templates.extend(pool.get(key, []))
        elif blueprint.blueprint_source == "profile:ugc_net_cs":
            selected_templates.extend(
                self._round_robin_template_groups(
                    [pool.get(key, []) for key in ("data", "algorithm", "database", "operating", "network", "theory", "discrete")],
                    max(request.question_count + 8, 24),
                )
            )

        if len(self._dedupe_templates(selected_templates)) < request.question_count:
            for key, templates in pool.items():
                if blueprint.subject == "Cyber Security" and key != "cyber":
                    continue
                if key == "science" and "school" not in unit_text and self._normalize(blueprint.subject) != "science":
                    continue
                if key == "data" and not re.search(r"\bdata\s+structures?\b", unit_text):
                    continue
                if self._contains_term(unit_text, key):
                    selected_templates.extend(templates)

        if not selected_templates:
            for unit in blueprint.syllabus_units:
                selected_templates.extend(self._templates_for_unit(unit, pool))

        selected_templates = self._dedupe_templates(selected_templates)
        selected_templates = self._rank_templates_for_blueprint(selected_templates, request, blueprint)
        if not selected_templates:
            return []

        normalized_templates = [self._template_with_difficulty(item) for item in selected_templates]
        buckets: dict[MockTestDifficulty, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": [], "mixed": []}
        for item in normalized_templates:
            difficulty = item["difficulty"]
            if difficulty in {"easy", "medium", "hard"}:
                buckets[difficulty].append(item)

        candidate_count = max(request.question_count, min(50, request.question_count + 6))
        desired_sequence = self._difficulty_sequence(request.difficulty, candidate_count)
        questions: list[MockQuestion] = []
        used_prompts: set[str] = set()
        bucket_positions: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0, "mixed": 0, "all": 0}
        for index, desired in enumerate(desired_sequence):
            candidate_groups = [
                (desired, buckets[desired]),
                ("medium", buckets["medium"]),
                ("easy", buckets["easy"]),
                ("hard", buckets["hard"]),
                ("all", normalized_templates),
            ]
            selected = self._next_unused_template(candidate_groups, bucket_positions, used_prompts)
            if not selected:
                break
            used_prompts.add(str(selected.get("prompt", "")).strip().lower())
            item = self._adapt_template_for_difficulty(selected, desired, blueprint, index)
            question = MockQuestion(
                id=f"q{index + 1}",
                prompt=str(item["prompt"]),
                options=list(item["options"]),
                correct_option_index=int(item["correct_option_index"]),
                explanation=str(item["explanation"]),
                difficulty=desired,
                tags=[str(tag)[:32] for tag in item.get("tags", [])][:4] or blueprint.syllabus_units[:2],
            )
            questions.append(question)
        return questions

    def generation_system_prompt(self) -> str:
        return (
            "You are Astra's exam-aware mock test generator. Return strict JSON only. "
            "Generate real subject questions, not study advice. Every MCQ must require applying syllabus concepts. "
            "Reject generic prompts like core purpose, prioritize first, practice habit, or clear fundamentals. "
            "Use exactly four meaningful options, one correct option, and a concept explanation. "
            "Honor the requested difficulty as a real cognitive level, not just a label. "
            "Do not wrap JSON in markdown fences. Keep each question and explanation compact."
        )

    def generation_user_prompt(
        self,
        request: MockTestGenerateRequest,
        blueprint: MockBlueprint,
        candidate_count: int | None = None,
        rejected_reasons: list[str] | None = None,
        accepted_count: int = 0,
    ) -> str:
        candidate_count = max(1, min(60, candidate_count or request.question_count))
        difficulty_sequence = self._difficulty_sequence(request.difficulty, candidate_count)
        return json.dumps(
            {
                "task": "generate_mock_test_questions",
                "topic": request.topic,
                "exam": blueprint.exam,
                "subject": blueprint.subject,
                "generation_mode": blueprint.generation_mode,
                "target_question_count": request.question_count,
                "candidate_count": candidate_count,
                "accepted_so_far": accepted_count,
                "difficulty": request.difficulty,
                "constraints": request.constraints,
                "difficulty_policy": self._difficulty_policy(request.difficulty),
                "difficulty_sequence": difficulty_sequence,
                "mode": "mcq",
                "repair_context": {
                    "rejected_reasons": (rejected_reasons or [])[:12],
                    "instruction": (
                        "If rejected_reasons are present, generate fresh replacement candidates that directly fix those issues. "
                        "If candidate_count is 1, return exactly one compact question in the questions array."
                    ),
                },
                "blueprint": {
                    "source": blueprint.blueprint_source,
                    "confidence": blueprint.confidence,
                    "syllabus_units": blueprint.syllabus_units,
                    "question_style": blueprint.question_style,
                    "expected_terms": blueprint.expected_terms[:40],
                    "banned_patterns": [
                        "Which option best describes the core purpose",
                        "What should you prioritize first",
                        "Which habit improves performance",
                        "Clear fundamentals",
                    ],
                },
                "quality_rules": [
                    "Generate more candidates than the final test needs; weak candidates will be discarded.",
                    "Return compact valid JSON only; no markdown fences and no prose outside JSON.",
                    "Keep each prompt under 90 words and each explanation under 45 words.",
                    "Every question must be domain-specific for the topic and use concepts from expected_terms or close synonyms.",
                    *self._constraint_rules(request),
                    "For programming/coding topics, prefer short code-output, bug-finding, API behavior, and edge-case MCQs.",
                    "Do not create study-advice, preparation strategy, or generic concept-purpose questions.",
                    "Use the difficulty_sequence in order. For hard items, use multi-step reasoning, code tracing, edge cases, or close distractors.",
                ],
                "schema": {
                    "questions": [
                        {
                            "prompt": "domain-specific question text",
                            "options": ["A", "B", "C", "D"],
                            "correct_option_index": 0,
                            "explanation": "why the answer is correct",
                            "difficulty": "easy|medium|hard",
                            "tags": ["syllabus unit", "concept"],
                        }
                    ]
                },
            },
            ensure_ascii=True,
        )

    async def _blueprint_from_llm(self, request: MockTestGenerateRequest) -> tuple[MockBlueprint | None, list[str]]:
        if not self.llm.model_configured():
            return None, [self.llm.missing_setup_for_model()]
        system_prompt = (
            "You are Astra's exam syllabus planner. Return strict JSON only. "
            "Infer the exam/topic syllabus and pattern for MCQ practice. Do not generate questions here. "
            "For programming topics, include language-specific runtime concepts, APIs, syntax rules, debugging patterns, and coding edge cases. "
            "For mathematics or numerical constraints, include computation-oriented units and terms such as counting, probability, recurrence, graph degree, subsets, permutations, and combinations. "
            "expected_terms must contain concrete domain terms that should appear in valid questions or explanations."
        )
        user_prompt = json.dumps(
            {
                "topic": request.topic,
                "constraints": request.constraints,
                "source_query": request.source_query,
                "difficulty": request.difficulty,
                "schema": {
                    "exam": "exam name or empty",
                    "subject": "subject or empty",
                    "syllabus_units": ["unit"],
                    "question_style": "short style description",
                    "expected_terms": ["concept term"],
                    "confidence": 0.0,
                },
            },
            ensure_ascii=True,
        )
        raw, setup = await self.llm.complete(system_prompt, user_prompt)
        payload = self._parse_json(raw)
        if setup or not payload:
            return None, setup
        units = self._clean_list(payload.get("syllabus_units"), 16)
        terms = self._clean_list(payload.get("expected_terms"), 50)
        if not units or not terms:
            return None, []
        terms = self._merge_terms(terms, self._terms(" ".join([request.topic, str(payload.get("subject") or ""), *units])))
        return (
            MockBlueprint(
                exam=str(payload.get("exam") or "").strip()[:80],
                subject=str(payload.get("subject") or "").strip()[:80],
                topic=request.topic,
                generation_mode="llm_planned",
                blueprint_source="llm_planner",
                syllabus_units=units,
                question_style=str(payload.get("question_style") or "Exam-pattern conceptual MCQ.")[:240],
                expected_terms=terms,
                confidence=self._float(payload.get("confidence"), 0.74),
            ),
            [],
        )

    def _blueprint_from_profile(self, profile: ExamProfile, request: MockTestGenerateRequest) -> MockBlueprint:
        is_pyq_style = self._is_pyq_style(request)
        generation_mode = "pyq_style" if is_pyq_style else "profile_based"
        focus_units = self._focus_units(request.topic, profile.syllabus_units)
        return MockBlueprint(
            exam=request.exam or profile.exam,
            subject=profile.subject or request.subject,
            topic=request.topic,
            generation_mode=generation_mode,
            blueprint_source=f"profile:{profile.id}",
            syllabus_units=focus_units or list(profile.syllabus_units[:8]),
            question_style=profile.question_style,
            expected_terms=list(profile.expected_terms),
            confidence=0.95,
        )

    def _blueprint_from_uploaded_syllabus(self, request: MockTestGenerateRequest) -> MockBlueprint | None:
        if not self.documents:
            return None
        terms = self._terms(f"{request.topic} {request.source_query}")
        best_excerpt = ""
        best_record: Any = None
        best_score = 0
        for record in self.documents.list_documents():
            excerpt = self.documents.document_excerpt(record.id, max_chars=5000)
            score = sum(excerpt.lower().count(term) for term in terms)
            if score > best_score:
                best_score = score
                best_excerpt = excerpt
                best_record = record
        if not best_record or best_score <= 0:
            return None
        units = self._units_from_text(best_excerpt, request.topic)
        if not units:
            return None
        return MockBlueprint(
            topic=request.topic,
            generation_mode="syllabus_based",
            blueprint_source="uploaded_syllabus",
            syllabus_units=units,
            expected_terms=self._terms(" ".join(units)),
            confidence=0.82,
            sources=[Source(title=best_record.title, url=f"document:{best_record.id}", snippet=best_record.text_preview, provider="Uploaded PDF")],
            source_text=best_excerpt[:5000],
        )

    def _match_profile(self, request: MockTestGenerateRequest) -> ExamProfile | None:
        haystack = self._normalize(" ".join([request.topic, request.exam, request.subject, request.source_query, *request.constraints]))
        best: tuple[int, ExamProfile] | None = None
        for profile in self.profiles:
            score = 0
            for alias in profile.aliases:
                normalized_alias = self._normalize(alias)
                if not self._profile_alias_allowed(profile, normalized_alias, haystack):
                    continue
                if normalized_alias and normalized_alias in haystack:
                    score = max(score, len(normalized_alias))
            if score and (best is None or score > best[0]):
                best = (score, profile)
        return best[1] if best else None

    def _profile_alias_allowed(self, profile: ExamProfile, normalized_alias: str, haystack: str) -> bool:
        if profile.id.startswith("school_") and re.search(r"\b(jee|neet|gate|ugc|net|upsc|ssc|banking|cgl)\b", haystack):
            return bool(re.search(r"\b(school|class|grade|standard|std|cbse|icse)\b", haystack))
        if profile.id == "school_science" and normalized_alias == "science":
            if re.search(r"\b(computer science|data science|political science|discrete mathematics)\b", haystack):
                return False
            return bool(re.search(r"\b(school|class 9|class 10|science basics|science quiz|science test)\b", haystack))
        if profile.id == "school_math" and normalized_alias == "mathematics basics":
            return bool(re.search(r"\b(school|class 9|class 10|basics)\b", haystack))
        if profile.id == "indian_history" and normalized_alias == "history":
            if re.search(r"\b(python|javascript|programming|computer|browser|git|code|software)\b", haystack):
                return False
        if profile.id == "ugc_net_paper1":
            paper_two_or_subject_specific = re.search(
                r"\b(paper\s*(?:2|ii)|paper[- ]?(?:2|ii)|computer\s+science|computer\s+applications?|cs|cse|dsa|data\s+structures?|"
                r"algorithm|arrays?|dbms|operating\s+systems?|computer\s+networks?|history|geography|commerce|management|economics?|"
                r"education|english|hindi|sociology|psychology|law|political\s+science|library\s+science|environmental\s+science)\b",
                haystack,
            )
            explicit_paper_one = re.search(r"\b(paper\s*(?:1|i)|paper[- ]?(?:1|i)|teaching\s+aptitude|research\s+aptitude)\b", haystack)
            if paper_two_or_subject_specific and not explicit_paper_one:
                return False
        return True

    def _focus_units(self, topic: str, units: tuple[str, ...]) -> list[str]:
        normalized_topic = self._normalize(topic)
        shortcuts = {
            "dbms": ("database", "databases", "normalization", "sql"),
            "database": ("database", "databases", "normalization", "sql"),
            "os": ("operating", "scheduling", "deadlock", "paging"),
            "cn": ("network", "networks", "tcp", "subnet"),
            "toc": ("theory", "automata", "computation"),
            "dsa": ("data", "algorithm", "structures", "array", "stack", "queue", "tree", "graph"),
            "array": ("array", "data", "algorithm", "structures"),
            "arrays": ("array", "data", "algorithm", "structures"),
            "genetics": ("genetics", "evolution"),
        }
        expanded_topic = normalized_topic
        for key, expansions in shortcuts.items():
            if key in normalized_topic:
                expanded_topic = f"{expanded_topic} {' '.join(expansions)}"
        broad_terms = {"computer", "science", "ugc", "net", "gate", "cse", "cs"}
        matches = [unit for unit in units if any(self._contains_term(normalized_topic, term) for term in self._terms(unit) if term not in broad_terms)]
        if not matches:
            matches = [unit for unit in units if any(self._contains_term(expanded_topic, term) for term in self._terms(unit) if term not in broad_terms)]
        return matches[:6]

    def _templates_for_unit(self, unit: str, pool: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        normalized = self._normalize(unit)
        templates: list[dict[str, Any]] = []
        for key, items in pool.items():
            if self._contains_term(normalized, key) or any(self._contains_term(self._normalize(tag), key) for item in items for tag in item.get("tags", [])):
                templates.extend(items)
        return templates

    def _next_unused_template(
        self,
        candidate_groups: list[tuple[str, list[dict[str, Any]]]],
        bucket_positions: dict[str, int],
        used_prompts: set[str],
    ) -> dict[str, Any] | None:
        for bucket_name, candidates in candidate_groups:
            if not candidates:
                continue
            start = bucket_positions.get(bucket_name, 0)
            for offset in range(len(candidates)):
                item = candidates[(start + offset) % len(candidates)]
                prompt_key = str(item.get("prompt", "")).strip().lower()
                if prompt_key and prompt_key not in used_prompts:
                    bucket_positions[bucket_name] = start + offset + 1
                    return item
        return None

    def _round_robin_template_groups(self, groups: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        max_len = max((len(group) for group in groups), default=0)
        for index in range(max_len):
            for group in groups:
                if index < len(group):
                    merged.append(group[index])
                    if len(merged) >= limit:
                        return merged
        return merged

    def _focused_template_keys(self, topic: str, pool: dict[str, list[dict[str, Any]]]) -> list[str]:
        normalized = self._normalize(topic)
        pattern_map: tuple[tuple[str, tuple[str, ...]], ...] = (
            (r"\barrays?\b", ("array",)),
            (r"\b(?:dbms|databases?|normalization|sql|transactions?)\b", ("database",)),
            (r"\b(?:operating\s+systems?|os|deadlock|paging|scheduling|memory\s+management)\b", ("operating",)),
            (r"\b(?:computer\s+networks?|networking|network|cn|tcp|udp|subnet|routing)\b", ("network",)),
            (r"\b(?:toc|automata|theory\s+of\s+computation|regular\s+languages?|pda|cfg|compiler|lexical|lexer|parser)\b", ("theory",)),
            (r"\b(?:discrete|combinatorics|set\s+theory|relations?|graph\s+theory)\b", ("discrete",)),
            (r"\b(?:python|django|flask|pandas)\b", ("python",)),
            (r"\b(?:genetics|mendel|dna|rna|inheritance)\b", ("genetics",)),
            (r"\b(?:electricity|current|resistance|voltage|ohm|optics|motion|force|electrostatics?|electric\s+field|potential|capacitance)\b", ("physics",)),
            (r"\b(?:chemistry|acid|base|mole|bond|enthalpy)\b", ("chemistry",)),
            (r"\b(?:commerce|accounting|audit|auditing|income\s+tax|gst|business\s+finance|financial\s+management)\b", ("commerce",)),
            (r"\b(?:history|modern\s+india|ancient\s+india|medieval\s+india)\b", ("history",)),
            (r"\b(?:geography|climate|monsoon|geomorphology)\b", ("geography",)),
            (r"\bteaching\b", ("teaching",)),
            (r"\bresearch\b", ("research",)),
            (r"\bcommunication\b", ("communication",)),
            (r"\b(?:logical|syllogism|fallacy)\b", ("logical",)),
            (r"\b(?:mathematical|ratio|percentage|average)\b", ("mathematical",)),
            (r"\b(?:interpretation|bar\s+chart|pie\s+chart)\b", ("interpretation",)),
            (r"\bict\b", ("ict",)),
            (r"\b(?:environment|environmental)\b", ("environment",)),
            (r"\b(?:higher\s+education|universit(?:y|ies)|ugc\s+policy)\b", ("higher",)),
            (r"\b(?:polity|constitution|political|fundamental\s+rights?|directive\s+principles|parliament|judiciary)\b", ("polity",)),
            (r"\bcyber\b", ("cyber",)),
        )
        keys: list[str] = []
        for pattern, candidates in pattern_map:
            if re.search(pattern, normalized):
                for key in candidates:
                    if key in pool and key not in keys:
                        keys.append(key)
        return keys

    def _units_from_text(self, text: str, topic: str) -> list[str]:
        if not text.strip():
            return []
        candidates = re.findall(r"(?:Unit|Paper|Section|Chapter)\s*[-IVX0-9A-Z]*\s*[:.-]\s*([A-Za-z][A-Za-z0-9 /,&()-]{8,90})", text)
        if len(candidates) < 3:
            candidates.extend(re.findall(r"\b([A-Z][A-Za-z]+(?:\s+(?:and|of|in|[A-Z][A-Za-z0-9]+)){1,5})\b", text[:2500]))
        cleaned: list[str] = []
        for candidate in candidates:
            item = re.sub(r"\s+", " ", candidate).strip(" .:-")
            if 4 <= len(item) <= 90 and item.lower() not in {topic.lower(), "table of contents"}:
                cleaned.append(item)
        unique = []
        seen: set[str] = set()
        for item in cleaned:
            key = item.lower()
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return unique[:12]

    def _is_pyq_style(self, request: MockTestGenerateRequest) -> bool:
        combined = self._normalize(" ".join([request.topic, request.source_query, *request.constraints]))
        return "pyq style" in combined or "pyqstyle" in combined or "previous year style" in combined

    def _terms(self, text: str) -> list[str]:
        stop = {
            "mock",
            "test",
            "exam",
            "practice",
            "question",
            "questions",
            "syllabus",
            "paper",
            "unit",
            "and",
            "the",
            "for",
            "with",
            "from",
            "only",
        }
        return [term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9+.-]{2,}", text.lower()) if term not in stop][:60]

    def _merge_terms(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                clean = str(item).strip()
                key = self._normalize(clean)
                if not clean or not key or key in seen:
                    continue
                merged.append(clean)
                seen.add(key)
                if len(merged) >= 60:
                    return merged
        return merged

    def _clean_list(self, raw: Any, limit: int) -> list[str]:
        if not isinstance(raw, list):
            return []
        return [str(item).strip()[:90] for item in raw if str(item).strip()][:limit]

    def _float(self, raw: Any, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            return default

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

    def _normalize(self, value: str) -> str:
        return re.sub(r"[^a-z0-9+]+", " ", value.lower()).strip()

    def _contains_term(self, text: str, term: str) -> bool:
        clean_text = self._normalize(text)
        clean_term = self._normalize(term)
        if not clean_term:
            return False
        pattern = rf"\b{re.escape(clean_term)}(?:s|es)?\b"
        return bool(re.search(pattern, clean_text))

    def _dedupe_templates(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            prompt = str(item.get("prompt", "")).lower()
            if prompt and prompt not in seen:
                unique.append(item)
                seen.add(prompt)
        return unique

    def _rank_templates_for_blueprint(self, items: list[dict[str, Any]], request: MockTestGenerateRequest, blueprint: MockBlueprint) -> list[dict[str, Any]]:
        if not items:
            return []
        if blueprint.blueprint_source == "profile:ugc_net_cs" and not re.search(
            r"\b(dbms|database|os|operating|network|toc|automata|compiler|discrete|algorithm|data\s+structure|arrays?)\b",
            self._normalize(request.topic),
        ):
            return items
        terms = [
            term
            for term in self._merge_terms(
                self._terms(request.topic),
                self._terms(" ".join(blueprint.syllabus_units)),
                list(blueprint.expected_terms),
            )
            if len(self._normalize(term)) >= 4
        ]
        if not terms:
            return items

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, item in enumerate(items):
            combined = self._normalize(
                " ".join(
                    [
                        str(item.get("prompt", "")),
                        str(item.get("explanation", "")),
                        " ".join(str(option) for option in item.get("options", [])),
                        " ".join(str(tag) for tag in item.get("tags", [])),
                    ]
                )
            )
            score = sum(1 for term in terms if self._contains_term(combined, term))
            scored.append((score, index, item))

        focused = [item for score, _, item in scored if score > 0]
        if len(focused) >= min(len(items), max(3, request.question_count)):
            return [item for _, _, item in sorted((entry for entry in scored if entry[0] > 0), key=lambda entry: (-entry[0], entry[1]))]
        return [item for _, _, item in sorted(scored, key=lambda entry: (-entry[0], entry[1]))]

    def _difficulty_sequence(self, difficulty: MockTestDifficulty, count: int) -> list[MockTestDifficulty]:
        if difficulty in {"easy", "medium", "hard"}:
            return [difficulty] * max(1, count)
        pattern: list[MockTestDifficulty] = ["easy", "medium", "medium", "hard", "easy", "medium", "hard", "medium", "easy", "hard"]
        return [pattern[index % len(pattern)] for index in range(max(1, count))]

    def _difficulty_policy(self, difficulty: MockTestDifficulty) -> dict[str, str]:
        policies = {
            "easy": "Direct concept checks: definitions, simple facts, one-step code/output, no traps, no multi-step reasoning.",
            "medium": "Application checks: compare concepts, apply one rule, interpret a short scenario, or solve a small calculation/code trace.",
            "hard": "Reasoning checks: multi-step scenarios, edge cases, code tracing, subtle distinctions, calculations, or elimination between close options.",
            "mixed": "Use the provided difficulty_sequence exactly, mixing direct, application, and harder reasoning questions.",
        }
        return {
            "requested": difficulty,
            "easy": policies["easy"],
            "medium": policies["medium"],
            "hard": policies["hard"],
            "instruction": policies[difficulty],
        }

    def _constraint_rules(self, request: MockTestGenerateRequest) -> list[str]:
        combined = self._normalize(" ".join(request.constraints))
        rules: list[str] = []
        if re.search(r"\b(numerical|numeric|calculation|calculate|quantitative|problem solving)\b", combined):
            rules.append(
                "The user requested numerical-only questions: each prompt must include concrete values and ask to calculate, find, determine, count, or choose a numeric result."
            )
            rules.append("For numerical-only questions, avoid pure definition/truth-value items and make at least two answer options numeric or formula-like.")
        return rules

    def _template_with_difficulty(self, item: dict[str, Any]) -> dict[str, Any]:
        clean = dict(item)
        difficulty = str(clean.get("difficulty") or "auto").strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = self._infer_template_difficulty(clean)
        clean["difficulty"] = difficulty
        return clean

    def _infer_template_difficulty(self, item: dict[str, Any]) -> MockTestDifficulty:
        prompt = str(item.get("prompt", ""))
        normalized = self._normalize(prompt)
        if re.search(
            r"\b(consider|given|if|after|trace|output|calculate|determine|schedule|serializable|dihybrid|independent assortment|b\+ tree|parent of index|edge case|which statement)\b",
            normalized,
        ) or "\n" in prompt:
            return "hard"
        if re.match(r"^\s*(what is|which term|who is|which body|which protocol|which keyword|a codon consists)\b", prompt, flags=re.IGNORECASE):
            return "easy"
        return "medium"

    def _adapt_template_for_difficulty(self, item: dict[str, Any], desired: MockTestDifficulty, blueprint: MockBlueprint, index: int) -> dict[str, Any]:
        adapted = dict(item)
        base_difficulty = str(item.get("difficulty") or "medium")
        if base_difficulty == desired:
            adapted["difficulty"] = desired
            return adapted

        subject = blueprint.subject or blueprint.topic or "this topic"
        prompt = str(adapted["prompt"])
        explanation = str(adapted["explanation"])
        if desired == "easy":
            adapted["prompt"] = self._easy_prompt(prompt)
            adapted["explanation"] = f"At easy difficulty, the key idea is direct recognition. {explanation}"
        elif desired == "medium":
            adapted["prompt"] = f"In a typical {subject} application, {prompt[0].lower() + prompt[1:] if prompt else prompt}"
            adapted["explanation"] = f"This medium item asks you to apply the concept, not only recall it. {explanation}"
        elif desired == "hard":
            adapted["prompt"] = (
                f"Consider this {subject} edge case #{index + 1}: {prompt} "
                "Which option remains correct after applying the relevant rule carefully?"
            )
            adapted["explanation"] = f"This hard item requires careful application and elimination between close choices. {explanation}"

        adapted["difficulty"] = desired
        tags = [str(tag) for tag in adapted.get("tags", []) if str(tag).strip()]
        adapted["tags"] = [*tags[:3], desired]
        return adapted

    def _easy_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            return prompt
        prompt = re.sub(r"^\s*(consider|given|if|after)\b[:,]?\s*", "", prompt, flags=re.IGNORECASE)
        return f"Direct concept check: {prompt[0].lower() + prompt[1:] if prompt else prompt}"

    def _profiles(self) -> list[ExamProfile]:
        return [
            ExamProfile(
                id="ugc_net_paper1",
                exam="UGC NET",
                subject="Paper I: Teaching and Research Aptitude",
                aliases=("ugc net", "net ugc", "nta net", "ugc net paper 1", "ugc net paper i", "ugc net general", "net paper 1"),
                syllabus_units=(
                    "Teaching Aptitude",
                    "Research Aptitude",
                    "Comprehension",
                    "Communication",
                    "Mathematical Reasoning and Aptitude",
                    "Logical Reasoning",
                    "Data Interpretation",
                    "Information and Communication Technology",
                    "People, Development and Environment",
                    "Higher Education System",
                ),
                question_style="UGC NET Paper I MCQs on teaching, research, reasoning, ICT, environment, and higher education.",
                expected_terms=(
                    "teaching",
                    "learner",
                    "evaluation",
                    "research",
                    "hypothesis",
                    "sampling",
                    "communication",
                    "barrier",
                    "reasoning",
                    "syllogism",
                    "fallacy",
                    "data",
                    "interpretation",
                    "percentage",
                    "average",
                    "ratio",
                    "work",
                    "ict",
                    "environment",
                    "sustainable",
                    "higher education",
                    "university",
                    "assessment",
                    "bar chart",
                ),
            ),
            ExamProfile(
                id="ugc_net_cs",
                exam="UGC NET",
                subject="Computer Science",
                aliases=(
                    "ugc net cs",
                    "net ugc cs",
                    "ugc net computer science",
                    "net computer science",
                    "ugc net cse",
                    "ugc net computer applications",
                    "ugc net dsa",
                    "ugc net data structures",
                    "ugc net data structure and algorithms",
                    "ugc net algorithms",
                    "ugc net arrays",
                    "ugc net dbms",
                    "ugc net operating system",
                    "ugc net computer networks",
                    "ugc net theory of computation",
                    "ugc net compiler",
                    "net paper 2 computer science",
                    "net paper ii computer science",
                ),
                syllabus_units=(
                    "Discrete Mathematics and Optimization",
                    "Data Structures and Algorithms",
                    "Programming in C and C++",
                    "Database Management Systems",
                    "Operating System",
                    "Computer Networks",
                    "Theory of Computation and Compilers",
                    "Computer Organization and Architecture",
                    "Software Engineering",
                    "Artificial Intelligence",
                ),
                question_style="UGC NET Paper II conceptual MCQs with theory, definitions, and applied CS reasoning.",
                expected_terms=(
                    "algorithm",
                    "complexity",
                    "array",
                    "row-major",
                    "address",
                    "index",
                    "stack",
                    "queue",
                    "tree",
                    "graph",
                    "heap",
                    "hash",
                    "normalization",
                    "transaction",
                    "deadlock",
                    "paging",
                    "tcp",
                    "dfa",
                    "compiler",
                    "cache",
                    "heuristic",
                    "predicate",
                    "permutation",
                    "combination",
                    "inclusion-exclusion",
                    "pigeonhole",
                    "recurrence",
                    "boolean",
                    "minterm",
                    "degree",
                ),
            ),
            ExamProfile(
                id="ugc_net_commerce",
                exam="UGC NET",
                subject="Commerce",
                aliases=(
                    "ugc net commerce",
                    "net commerce",
                    "ugc net paper 2 commerce",
                    "ugc net accounting",
                    "ugc net business finance",
                    "ugc net income tax",
                    "ugc net auditing",
                ),
                syllabus_units=(
                    "Accounting and Auditing",
                    "Business Finance",
                    "Income Tax and Corporate Tax Planning",
                    "Business Economics",
                    "Business Management",
                    "Marketing Management",
                    "Human Resource Management",
                    "Banking and Financial Institutions",
                    "International Business",
                ),
                question_style="UGC NET Commerce Paper II MCQs with accounting, audit, tax, finance, and management application.",
                expected_terms=(
                    "accounting",
                    "standard",
                    "audit",
                    "depreciation",
                    "ratio",
                    "working capital",
                    "tax",
                    "gst",
                    "cost",
                    "variance",
                    "management",
                    "marketing",
                    "finance",
                ),
            ),
            ExamProfile(
                id="ugc_net_political_science",
                exam="UGC NET",
                subject="Political Science",
                aliases=(
                    "ugc net political science",
                    "net political science",
                    "ugc net polity",
                    "ugc net paper 2 political science",
                    "ugc net fundamental rights",
                ),
                syllabus_units=(
                    "Political Theory",
                    "Indian Constitution",
                    "Fundamental Rights and Directive Principles",
                    "Union and State Government",
                    "Parliament and Judiciary",
                    "Comparative Politics",
                    "International Relations",
                    "Public Administration",
                ),
                question_style="UGC NET Political Science Paper II MCQs with constitution, theory, institutions, and statement-based reasoning.",
                expected_terms=(
                    "constitution",
                    "fundamental rights",
                    "article",
                    "parliament",
                    "judiciary",
                    "federalism",
                    "rights",
                    "directive principles",
                    "governor",
                    "political theory",
                ),
            ),
            ExamProfile(
                id="gate_cse",
                exam="GATE",
                subject="Computer Science",
                aliases=("gate cse", "gate cs", "gate computer science", "gate cse dbms", "gate dbms"),
                syllabus_units=(
                    "Engineering Mathematics",
                    "Digital Logic",
                    "Computer Organization and Architecture",
                    "Programming and Data Structures",
                    "Algorithms",
                    "Theory of Computation",
                    "Compiler Design",
                    "Operating System",
                    "Databases",
                    "Computer Networks",
                ),
                question_style="GATE CSE conceptual and numerical MCQs with precise CS reasoning.",
                expected_terms=(
                    "normalization",
                    "b+ tree",
                    "transaction",
                    "sql",
                    "key",
                    "relation",
                    "locking",
                    "concurrency",
                    "er model",
                    "deadlock",
                    "process",
                    "scheduling",
                    "paging",
                    "page fault",
                    "mutex",
                    "safe sequence",
                    "tcp",
                    "subnet",
                    "ipv4",
                    "arp",
                    "dns",
                    "osi",
                    "congestion",
                    "cache",
                    "dfa",
                    "complexity",
                    "pipeline",
                ),
            ),
            ExamProfile(
                id="cyber_security",
                exam="",
                subject="Cyber Security",
                aliases=("cyber security", "cybersecurity", "information security", "security numerical", "cyber security numerical"),
                syllabus_units=(
                    "Quantitative Risk Analysis",
                    "Cryptographic Mathematics",
                    "Password Entropy",
                    "Network Security Calculations",
                    "Vulnerability Scoring",
                    "Security Operations Metrics",
                ),
                question_style="Cyber security MCQs with quantitative risk, cryptography, entropy, and network-security calculations.",
                expected_terms=(
                    "cyber",
                    "security",
                    "risk",
                    "ale",
                    "sle",
                    "aro",
                    "rsa",
                    "modular",
                    "entropy",
                    "brute force",
                    "firewall",
                    "cvss",
                    "vulnerability",
                    "incident",
                    "availability",
                ),
            ),
            ExamProfile(
                id="dsa",
                exam="",
                subject="Data Structures and Algorithms",
                aliases=("dsa", "data structures", "algorithms", "coding interview", "interview dsa"),
                syllabus_units=("Arrays and Strings", "Linked Lists", "Stacks and Queues", "Trees", "Graphs", "Sorting", "Searching", "Dynamic Programming", "Greedy Algorithms", "Hashing"),
                question_style="Programming interview MCQs focused on complexity, invariants, and data-structure choice.",
                expected_terms=("array", "stack", "queue", "tree", "graph", "heap", "hash", "complexity", "binary search", "dynamic programming"),
            ),
            ExamProfile(
                id="discrete_math",
                exam="",
                subject="Discrete Mathematics",
                aliases=("discrete mathematics", "discrete math", "descrete mathematics", "descrete math", "discrete structures"),
                syllabus_units=(
                    "Counting Principles and Combinatorics",
                    "Set Theory and Functions",
                    "Relations and Partial Orders",
                    "Graph Theory",
                    "Recurrence Relations",
                    "Boolean Algebra",
                    "Number Theory and Modular Arithmetic",
                    "Discrete Probability",
                ),
                question_style="Discrete mathematics MCQs with numerical counting, graph, recurrence, Boolean, and modular arithmetic reasoning.",
                expected_terms=(
                    "permutation",
                    "combination",
                    "subset",
                    "inclusion-exclusion",
                    "pigeonhole",
                    "graph",
                    "degree",
                    "edge",
                    "tree",
                    "recurrence",
                    "boolean",
                    "minterm",
                    "modular",
                    "probability",
                ),
            ),
            ExamProfile(
                id="python_programming",
                exam="",
                subject="Python Programming",
                aliases=("python", "python programming", "python test", "python mock test", "python quiz", "programming in python", "python basics"),
                syllabus_units=(
                    "Python Syntax and Data Types",
                    "Control Flow",
                    "Functions and Scope",
                    "Lists Tuples Dictionaries and Sets",
                    "Comprehensions",
                    "Modules and Packages",
                    "File Handling",
                    "Exceptions",
                    "Object-Oriented Programming",
                    "Iterators and Generators",
                    "Decorators",
                    "Testing and Debugging",
                ),
                question_style="Python MCQs focused on syntax, runtime behavior, data structures, functions, OOP, and idiomatic code.",
                expected_terms=(
                    "python",
                    "list",
                    "dictionary",
                    "dict",
                    "tuple",
                    "set",
                    "function",
                    "lambda",
                    "exception",
                    "class",
                    "object",
                    "iterator",
                    "generator",
                    "decorator",
                    "module",
                    "comprehension",
                    "scope",
                    "mutable",
                    "immutable",
                ),
            ),
            ExamProfile(
                id="dbms",
                exam="",
                subject="DBMS",
                aliases=("dbms", "database management", "database systems", "sql", "normalization"),
                syllabus_units=("ER Model", "Relational Model", "SQL", "Normalization", "Transactions", "Indexing", "Concurrency Control", "Recovery"),
                question_style="Database MCQs on design, SQL, transactions, indexing, and normalization.",
                expected_terms=("relation", "key", "normalization", "sql", "transaction", "acid", "index", "serializable", "b+ tree", "locking", "concurrency", "er model"),
            ),
            ExamProfile(
                id="os",
                exam="",
                subject="Operating Systems",
                aliases=("operating system", "operating systems", "os", "process scheduling", "deadlock"),
                syllabus_units=("Processes and Threads", "CPU Scheduling", "Synchronization", "Deadlocks", "Memory Management", "Paging", "File Systems"),
                question_style="Operating system MCQs with scheduling, memory, synchronization, and deadlock reasoning.",
                expected_terms=("process", "thread", "semaphore", "deadlock", "paging", "page fault", "scheduling", "mutex"),
            ),
            ExamProfile(
                id="computer_networks",
                exam="",
                subject="Computer Networks",
                aliases=("computer networks", "computer network", "cn", "networking", "tcp ip"),
                syllabus_units=("OSI Model", "TCP/IP", "Routing", "Subnetting", "Data Link Layer", "Congestion Control", "Application Protocols"),
                question_style="Networking MCQs on protocols, layers, addressing, and routing behavior.",
                expected_terms=("tcp", "udp", "ip", "subnet", "routing", "osi", "ethernet", "congestion", "dns"),
            ),
            ExamProfile(
                id="toc",
                exam="",
                subject="Theory of Computation",
                aliases=("theory of computation", "toc", "automata", "compiler", "formal languages"),
                syllabus_units=("Finite Automata", "Regular Languages", "Context-Free Grammars", "Pushdown Automata", "Turing Machines", "Decidability", "Compiler Basics"),
                question_style="Formal language MCQs involving automata, grammars, decidability, and compiler basics.",
                expected_terms=("dfa", "nfa", "regular", "cfg", "pda", "turing", "decidable", "parser", "grammar"),
            ),
            ExamProfile(
                id="school_physics",
                exam="School",
                subject="Physics",
                aliases=(
                    "school physics",
                    "class 9 physics",
                    "class 10 physics",
                    "class 11 physics",
                    "class 12 physics",
                    "cbse physics",
                    "physics electricity",
                    "current electricity",
                    "electricity",
                    "motion physics",
                    "light physics",
                ),
                syllabus_units=("Motion", "Force and Laws of Motion", "Work and Energy", "Sound", "Light", "Electricity", "Magnetism"),
                question_style="School Physics MCQs with textbook concepts, formula application, and clean numerical reasoning.",
                expected_terms=("physics", "force", "acceleration", "energy", "current", "voltage", "resistance", "ohm", "power", "lens", "mirror", "charge", "circuit"),
            ),
            ExamProfile(
                id="school_chemistry",
                exam="School",
                subject="Chemistry",
                aliases=("school chemistry", "class 9 chemistry", "class 10 chemistry", "cbse chemistry", "chemical reactions", "acids bases salts", "carbon compounds"),
                syllabus_units=("Matter", "Atoms and Molecules", "Chemical Reactions", "Acids Bases and Salts", "Metals and Non-metals", "Carbon and Compounds"),
                question_style="School Chemistry MCQs with reactions, definitions, and simple application.",
                expected_terms=("chemistry", "atom", "molecule", "reaction", "acid", "base", "salt", "metal", "carbon", "valency", "oxidation"),
            ),
            ExamProfile(
                id="school_biology",
                exam="School",
                subject="Biology",
                aliases=("school biology", "class 9 biology", "class 10 biology", "cbse biology", "life processes", "photosynthesis", "human biology"),
                syllabus_units=("Cell Biology", "Life Processes", "Photosynthesis", "Respiration", "Control and Coordination", "Reproduction", "Heredity"),
                question_style="School Biology MCQs with textbook facts plus process understanding.",
                expected_terms=("biology", "cell", "photosynthesis", "respiration", "enzyme", "stomata", "xylem", "phloem", "hormone", "reproduction", "heredity"),
            ),
            ExamProfile(
                id="jee_physics",
                exam="JEE",
                subject="Physics",
                aliases=("jee physics", "iit physics", "jee mains physics", "jee advanced physics", "jee current electricity", "jee electricity", "jee mechanics", "jee optics"),
                syllabus_units=("Mechanics", "Thermodynamics", "Electrostatics", "Current Electricity", "Magnetism", "Optics", "Modern Physics"),
                question_style="JEE Physics MCQs with formula application and conceptual reasoning.",
                expected_terms=("force", "acceleration", "potential", "current", "lens", "momentum", "energy", "field", "charge", "capacitance", "electrostatics", "wavelength"),
            ),
            ExamProfile(
                id="jee_math",
                exam="JEE",
                subject="Mathematics",
                aliases=("jee math", "jee mathematics", "iit math", "jee maths"),
                syllabus_units=("Algebra", "Trigonometry", "Coordinate Geometry", "Calculus", "Vectors", "Probability", "Complex Numbers"),
                question_style="JEE Mathematics MCQs with symbolic manipulation and problem solving.",
                expected_terms=("function", "derivative", "integral", "matrix", "probability", "vector", "complex", "limit"),
            ),
            ExamProfile(
                id="jee_chemistry",
                exam="JEE",
                subject="Chemistry",
                aliases=("jee chemistry", "iit chemistry", "jee mains chemistry"),
                syllabus_units=("Physical Chemistry", "Organic Chemistry", "Inorganic Chemistry", "Chemical Bonding", "Equilibrium", "Thermodynamics"),
                question_style="JEE Chemistry MCQs combining concepts, reactions, and calculations.",
                expected_terms=("mole", "bond", "equilibrium", "enthalpy", "orbital", "reaction", "acid", "base"),
            ),
            ExamProfile(
                id="neet",
                exam="NEET",
                subject="Physics, Chemistry, and Biology",
                aliases=("neet", "neet ug", "neet mock test", "neet practice test", "neet exam"),
                syllabus_units=(
                    "Biology",
                    "Genetics and Evolution",
                    "Human Physiology",
                    "Physics",
                    "Mechanics",
                    "Current Electricity",
                    "Chemistry",
                    "Chemical Bonding",
                    "Organic Chemistry",
                ),
                question_style="NEET mixed-subject MCQs across Biology, Physics, and Chemistry with NCERT-style concept checks.",
                expected_terms=(
                    "biology",
                    "gene",
                    "dna",
                    "mendel",
                    "cell",
                    "physiology",
                    "physics",
                    "force",
                    "current",
                    "resistance",
                    "energy",
                    "chemistry",
                    "mole",
                    "bond",
                    "atom",
                    "reaction",
                ),
            ),
            ExamProfile(
                id="neet_biology",
                exam="NEET",
                subject="Biology",
                aliases=("neet biology", "neet bio", "neet genetics", "biology genetics", "genetics"),
                syllabus_units=("Cell Biology", "Genetics and Evolution", "Human Physiology", "Plant Physiology", "Ecology", "Biotechnology"),
                question_style="NEET Biology fact-plus-concept MCQs aligned with NCERT style.",
                expected_terms=(
                    "gene",
                    "allele",
                    "chromosome",
                    "dna",
                    "rna",
                    "codon",
                    "hormone",
                    "enzyme",
                    "ecosystem",
                    "mendel",
                    "cross",
                    "inheritance",
                    "meiosis",
                    "dominance",
                    "reproduction",
                    "flower",
                    "photosynthesis",
                    "cell",
                    "blood",
                    "respiration",
                ),
            ),
            ExamProfile(
                id="neet_physics",
                exam="NEET",
                subject="Physics",
                aliases=("neet physics", "medical physics"),
                syllabus_units=("Mechanics", "Heat", "Waves", "Optics", "Electrostatics", "Current Electricity", "Modern Physics"),
                question_style="NEET Physics MCQs with direct formulas and conceptual clarity.",
                expected_terms=("velocity", "force", "current", "lens", "resistance", "energy", "frequency", "charge"),
            ),
            ExamProfile(
                id="neet_chemistry",
                exam="NEET",
                subject="Chemistry",
                aliases=("neet chemistry", "medical chemistry"),
                syllabus_units=("Basic Concepts", "Atomic Structure", "Chemical Bonding", "Equilibrium", "Organic Chemistry", "Biomolecules"),
                question_style="NEET Chemistry MCQs aligned to NCERT facts and concept application.",
                expected_terms=("mole", "bond", "atom", "equilibrium", "reaction", "isomer", "protein", "organic"),
            ),
            ExamProfile(
                id="ssc_banking_aptitude",
                exam="SSC/Banking",
                subject="Quantitative Aptitude and Reasoning",
                aliases=("ssc cgl", "banking reasoning", "banking aptitude", "quantitative aptitude", "reasoning", "syllogism"),
                syllabus_units=("Number System", "Percentage", "Ratio and Proportion", "Time and Work", "Syllogism", "Seating Arrangement", "Data Interpretation"),
                question_style="Aptitude MCQs with short calculations and reasoning elimination.",
                expected_terms=("percentage", "ratio", "profit", "work", "syllogism", "conclusion", "arrangement", "average"),
            ),
            ExamProfile(
                id="indian_history",
                exam="",
                subject="Indian History",
                aliases=("indian history", "modern history", "history", "class 10 history", "school history", "freedom struggle"),
                syllabus_units=("Revolt of 1857", "Indian National Congress", "Swadeshi Movement", "Gandhian Movements", "Civil Disobedience", "Quit India Movement"),
                question_style="Indian history MCQs focused on events, causes, leaders, and consequence-based reasoning.",
                expected_terms=("history", "congress", "movement", "gandhi", "swaraj", "swadeshi", "quit india", "civil disobedience", "rowlatt", "dandi"),
            ),
            ExamProfile(
                id="upsc_general_studies",
                exam="UPSC",
                subject="General Studies",
                aliases=("upsc polity", "upsc history", "upsc geography", "upsc current affairs", "civil services", "upsc fundamental rights", "polity fundamental rights"),
                syllabus_units=("Indian Polity", "Modern History", "Geography", "Economy", "Environment", "Science and Technology", "Current Affairs"),
                question_style="UPSC prelims-style MCQs with statement analysis and factual-conceptual links.",
                expected_terms=(
                    "constitution",
                    "article",
                    "parliament",
                    "monsoon",
                    "governor",
                    "movement",
                    "biodiversity",
                    "inflation",
                    "fundamental rights",
                    "rights",
                    "directive principles",
                    "judicial review",
                    "basic structure",
                    "polity",
                ),
            ),
            ExamProfile(
                id="school_science",
                exam="School",
                subject="Science",
                aliases=("school science", "class 10 science", "class 9 science", "science"),
                syllabus_units=("Motion", "Force", "Matter", "Atoms and Molecules", "Life Processes", "Electricity", "Light"),
                question_style="School science MCQs with textbook concepts and direct applications.",
                expected_terms=("force", "cell", "atom", "molecule", "current", "lens", "motion", "respiration"),
            ),
            ExamProfile(
                id="school_math",
                exam="School",
                subject="Mathematics",
                aliases=("school math", "class 10 math", "class 9 math", "mathematics basics"),
                syllabus_units=("Algebra", "Linear Equations", "Quadratic Equations", "Geometry", "Trigonometry", "Statistics", "Probability"),
                question_style="School mathematics MCQs with calculation and concept checks.",
                expected_terms=("equation", "triangle", "angle", "probability", "mean", "quadratic", "ratio", "graph"),
            ),
        ]

    def _question_pool(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "commerce": [
                self._q("Which accounting standard principle requires inventory to be valued at lower of cost and net realizable value?", ["Prudence principle", "Money measurement only", "Dual aspect only", "Separate entity only"], 0, "Prudence avoids overstating assets; inventory is carried at the lower of cost and net realizable value.", ["accounting", "standard", "inventory"], "medium"),
                self._q("A machine costs 100000, has residual value 10000, and useful life of 5 years. What is annual straight-line depreciation?", ["18000", "20000", "10000", "22000"], 0, "Straight-line depreciation is (cost - residual value) / useful life = (100000 - 10000) / 5 = 18000.", ["accounting", "depreciation", "commerce"], "medium"),
                self._q("Which audit evidence is generally considered more reliable?", ["Evidence obtained directly by the auditor from an external party", "Unverified oral explanation only", "A rough internal note", "A cancelled draft invoice"], 0, "External evidence obtained directly by the auditor has stronger reliability than unsupported internal explanations.", ["audit", "evidence", "commerce"], "medium"),
                self._q("Current ratio is calculated using which formula?", ["Current assets divided by current liabilities", "Net profit divided by sales", "Debt divided by equity", "Sales divided by inventory"], 0, "The current ratio measures short-term liquidity as current assets / current liabilities.", ["ratio", "liquidity", "commerce"], "easy"),
                self._q("If current assets are 300000 and current liabilities are 150000, what is the current ratio?", ["2:1", "1:2", "1.5:1", "3:1"], 0, "Current ratio equals 300000 / 150000 = 2, so it is 2:1.", ["ratio", "liquidity", "accounting"], "medium"),
                self._q("In cost accounting, material price variance mainly compares what?", ["Actual price with standard price for actual quantity", "Actual sales with budgeted sales", "Fixed cost with total cost", "Closing stock with opening stock"], 0, "Material price variance isolates the effect of paying a different price than the standard price for the actual quantity purchased or used.", ["cost accounting", "variance"], "hard"),
                self._q("Which tax is levied on supply of goods and services in India under the GST framework?", ["Goods and Services Tax", "Dividend Distribution Tax", "Wealth Tax", "Securities Transaction Tax only"], 0, "GST is a destination-based indirect tax on supplies of goods and services.", ["gst", "tax", "commerce"], "easy"),
                self._q("Working capital is best described by which expression?", ["Current assets minus current liabilities", "Fixed assets minus depreciation", "Sales minus gross profit", "Share capital plus reserves only"], 0, "Net working capital represents short-term operating liquidity: current assets less current liabilities.", ["working capital", "finance"], "medium"),
                self._q("Which capital budgeting method directly discounts future cash flows?", ["Net present value", "Payback period only", "Accounting rate of return", "Simple average profit"], 0, "NPV discounts expected cash flows at the required rate and subtracts initial investment.", ["finance", "npv", "capital budgeting"], "medium"),
                self._q("In break-even analysis, contribution is calculated as what?", ["Sales minus variable cost", "Sales minus fixed cost only", "Fixed cost divided by price", "Profit plus tax only"], 0, "Contribution is the amount left after variable costs to cover fixed costs and profit.", ["cost accounting", "break even"], "medium"),
                self._q("Consider a firm with sales 800000, variable cost 500000, and fixed cost 200000. What is the profit?", ["100000", "300000", "200000", "500000"], 0, "Profit equals sales - variable cost - fixed cost = 800000 - 500000 - 200000 = 100000.", ["cost accounting", "profit", "commerce"], "hard"),
                self._q("Which management function involves comparing actual performance with planned standards?", ["Controlling", "Staffing", "Directing only", "Forecasting only"], 0, "Controlling measures actual performance against standards and initiates corrective action.", ["management", "control"], "medium"),
            ],
            "cyber": [
                self._q("Calculate ALE when SLE is 5000 and ARO is 0.20 for a cyber security asset.", ["1000", "5200", "25000", "0.04"], 0, "Annualized Loss Expectancy is SLE x ARO, so 5000 x 0.20 = 1000.", ["cyber security", "risk", "ale"], "medium"),
                self._q("A firewall blocks 45 malicious requests out of 50. Determine the detection rate.", ["90%", "45%", "10%", "5%"], 0, "The firewall detection rate is 45 / 50 = 0.90, or 90%.", ["firewall", "cyber security", "security metrics"], "medium"),
                self._q("A password uses 8 symbols from a 16-character alphabet. Calculate its entropy.", ["32 bits", "16 bits", "64 bits", "128 bits"], 0, "Entropy is 8 x log2(16), and log2(16) = 4, so the password has 32 bits.", ["cyber security", "entropy", "password"], "medium"),
                self._q("A security service promises 99% availability over 1000 monitored minutes. Determine the allowed downtime.", ["10 minutes", "1 minute", "99 minutes", "100 minutes"], 0, "Downtime is 1% of 1000 minutes, which equals 10 minutes.", ["availability", "cyber security", "security metrics"], "medium"),
                self._q("An asset has SLE = 50000 and ARO = 0.20. A security control costs 8000 per year and lowers ARO to 0.05. What is the net annual value of the control?", ["-500", "2500", "7500", "8000"], 0, "Original ALE is 50000 x 0.20 = 10000. New ALE is 50000 x 0.05 = 2500, so savings are 7500 and net value is 7500 - 8000 = -500.", ["cyber security", "risk", "ale"], "hard"),
                self._q("For RSA, p = 17, q = 19, and public exponent e = 5. Determine the private exponent d modulo phi(n).", ["173", "115", "57", "283"], 0, "Here phi(n) = 16 x 18 = 288. The inverse of 5 modulo 288 is 173 because 5 x 173 = 865 = 3 x 288 + 1.", ["rsa", "modular", "cryptography"], "hard"),
                self._q("A password policy uses 12 independent characters from an alphabet of 62 symbols. Calculate the approximate password entropy.", ["71.45 bits", "62 bits", "48 bits", "744 bits"], 0, "Entropy is length x log2(alphabet size), so 12 x log2(62) is about 12 x 5.954 = 71.45 bits.", ["cyber security", "entropy", "password"], "hard"),
                self._q("A brute-force attack must search 2^40 keys and can test 2^20 keys per second. Calculate the expected full-search time.", ["2^20 seconds", "2^60 seconds", "20 seconds", "2^30 seconds"], 0, "Time equals keyspace divided by rate: 2^40 / 2^20 = 2^20 seconds, which is about 12.1 days.", ["brute force", "keyspace", "cyber security"], "hard"),
                self._q("A 64-bit security MAC is checked against 2^20 forged messages. Using the birthday approximation n^2 / 2^(b+1), calculate the collision probability.", ["About 2^-25", "About 2^-44", "About 2^-64", "About 2^-5"], 0, "With n = 2^20 and b = 64, n^2 / 2^(b+1) = 2^40 / 2^65 = 2^-25.", ["mac", "collision", "cyber security"], "hard"),
                self._q("A firewall drops 1800 malicious packets out of 2000 malicious packets and also blocks 100 legitimate packets out of 10000. Determine the false positive rate.", ["1%", "5%", "10%", "90%"], 0, "False positive rate uses legitimate traffic: 100 false positives out of 10000 legitimate packets equals 100 / 10000 = 1%.", ["firewall", "security metrics", "false positive"], "hard"),
                self._q("A vulnerability has exploitability 8 and impact 7. If a simplified risk score is exploitability x impact / 10, determine the score.", ["5.6", "15", "56", "1.14"], 0, "The simplified vulnerability score is 8 x 7 / 10 = 5.6.", ["vulnerability", "risk", "cvss"], "hard"),
                self._q("A system with 99.9% monthly availability is observed for a 30-day month. Approximately how many minutes of downtime are allowed?", ["43.2 minutes", "4.32 minutes", "432 minutes", "0.432 minutes"], 0, "A 30-day month has 30 x 24 x 60 = 43200 minutes. Downtime at 0.1% is 43.2 minutes.", ["availability", "security operations", "downtime"], "hard"),
                self._q("An IDS processes 50000 events and raises 250 alerts. Analysts confirm 40 true incidents. Determine what percentage of alerts were true positives.", ["16%", "8%", "0.08%", "84%"], 0, "True-positive alert percentage is confirmed incidents divided by alerts: 40 / 250 = 0.16, or 16%.", ["ids", "incident", "security metrics"], "hard"),
                self._q("A control reduces annual incident probability from 0.30 to 0.12 for an incident impact of 40000. If the control costs 5000 yearly, what is the annualized net benefit?", ["2200", "7200", "12000", "-5000"], 0, "Risk reduction savings are (0.30 - 0.12) x 40000 = 7200. Net benefit is 7200 - 5000 = 2200.", ["cyber security", "risk", "control"], "hard"),
                self._q("In RSA with p = 11 and q = 13, what is phi(n), the totient used for key generation?", ["120", "143", "24", "132"], 0, "For RSA, phi(n) = (p - 1)(q - 1) = 10 x 12 = 120.", ["rsa", "cryptography", "totient"], "hard"),
                self._q("A security scan finds 36 critical vulnerabilities in 900 hosts. What is the critical vulnerability density per 100 hosts?", ["4", "2.5", "25", "0.4"], 0, "Density per 100 hosts is 36 / 900 x 100 = 4 critical vulnerabilities per 100 hosts.", ["vulnerability", "security metrics", "density"], "hard"),
            ],
            "discrete": [
                self._q("How many distinct permutations can be formed from the letters of MATH?", ["24", "12", "16", "8"], 0, "MATH has four distinct letters, so the count is 4! = 24.", ["discrete mathematics", "permutation"], "easy"),
                self._q("How many 3-member committees can be formed from 8 people?", ["56", "24", "336", "11"], 0, "The number of committees is C(8,3) = 8 x 7 x 6 / (3 x 2 x 1) = 56.", ["combination", "counting"], "medium"),
                self._q("A set has 5 elements. How many subsets does it have?", ["32", "25", "10", "16"], 0, "A set with n elements has 2^n subsets, so 2^5 = 32.", ["set theory", "subset"], "easy"),
                self._q("In a class of 100 students, 60 take Math, 50 take Physics, and 20 take both. How many take neither subject?", ["10", "20", "30", "40"], 0, "By inclusion-exclusion, at least one subject is 60 + 50 - 20 = 90, so neither is 100 - 90 = 10.", ["inclusion-exclusion", "sets"], "medium"),
                self._q("What is the minimum number of people needed to guarantee that two share a birth month?", ["13", "12", "11", "24"], 0, "There are 12 months, so by the pigeonhole principle, 13 people guarantee a repeated month.", ["pigeonhole", "counting"], "easy"),
                self._q("A simple graph has vertex degrees 2, 3, 3, 4, and 4. How many edges does it have?", ["8", "16", "7", "9"], 0, "The degree sum is 16, and the handshaking lemma gives edges = 16 / 2 = 8.", ["graph", "degree"], "medium"),
                self._q("How many edges are in the complete graph K6?", ["15", "12", "18", "30"], 0, "A complete graph Kn has n(n-1)/2 edges, so K6 has 6 x 5 / 2 = 15.", ["graph", "complete graph"], "medium"),
                self._q("A tree has 11 vertices. How many edges must it have?", ["10", "11", "12", "22"], 0, "Every tree with n vertices has exactly n - 1 edges, so 11 vertices give 10 edges.", ["tree", "graph"], "easy"),
                self._q("For the recurrence a_n = 2a_(n-1) with a_0 = 3, what is a_4?", ["48", "24", "32", "16"], 0, "The sequence doubles each step: 3, 6, 12, 24, 48.", ["recurrence", "sequence"], "medium"),
                self._q("A Boolean function has 3 variables. How many minterms are possible?", ["8", "6", "3", "16"], 0, "Each of 3 Boolean variables has 2 choices, so the number of minterms is 2^3 = 8.", ["boolean algebra", "minterm"], "medium"),
                self._q("Which value of x satisfies 3x ≡ 1 (mod 7)?", ["5", "2", "3", "6"], 0, "3 x 5 = 15, and 15 mod 7 leaves remainder 1.", ["modular arithmetic", "inverse"], "hard"),
                self._q("A bag has 4 red and 6 blue balls. If 2 balls are drawn without replacement, what is the probability both are red?", ["2/15", "4/25", "1/5", "8/45"], 0, "The probability is (4/10) x (3/9) = 12/90 = 2/15.", ["probability", "combination"], "hard"),
                self._q("A connected planar graph has 6 vertices and 9 edges. How many faces does it have?", ["5", "3", "4", "6"], 0, "Euler's formula gives V - E + F = 2, so F = 2 - 6 + 9 = 5.", ["planar graph", "euler formula"], "hard"),
            ],
            "array": [
                self._q("For a 0-based one-dimensional array A with base address 1000 and element size 4 bytes, what is the address of A[5]?", ["1020", "1005", "1016", "1040"], 0, "The address is base + index x element size = 1000 + 5 x 4 = 1020.", ["array", "address", "index"], "medium"),
                self._q("In row-major storage of a 2D array A[10][20], which element lies immediately after A[3][19]?", ["A[4][0]", "A[3][20]", "A[2][19]", "A[4][19]"], 0, "Row-major order stores all columns of a row before moving to the next row, so A[4][0] follows A[3][19].", ["array", "row-major"], "medium"),
                self._q("Which operation is O(n) in the worst case for an unsorted static array?", ["Deleting an element after searching for its value", "Accessing A[i] by index", "Reading the first element", "Computing the base address"], 0, "The value must first be searched linearly; shifting may also be needed depending on order preservation.", ["array", "time complexity"], "medium"),
                self._q("Direct concept check: why does array indexing take O(1) time in a contiguous array?", ["The target address is computed arithmetically", "Every prior element is traversed", "A hash function is always used", "The array is stored as a linked list"], 0, "Contiguous storage allows base + offset calculation without scanning earlier elements.", ["array", "random access"], "easy"),
                self._q("For a sorted array of 31 elements, what is the maximum number of successful binary-search comparisons?", ["5", "31", "15", "6"], 0, "Binary search needs ceil(log2(31 + 1)) = 5 comparisons in the worst successful case.", ["array", "binary search", "complexity"], "medium"),
                self._q("Which situation makes insertion into an array at position k require shifting elements?", ["When order after position k must be preserved", "When only A[0] is read", "When element size is one byte", "When the array is already full of zeros"], 0, "To preserve order, elements from k onward must move right before the new item is placed.", ["array", "insertion"], "medium"),
                self._q("Consider an array implementation of a circular queue with front = 7, rear = 2, capacity = 10, using one empty slot to distinguish full from empty. How many elements are present?", ["5", "6", "4", "9"], 0, "The count is (rear - front + capacity) mod capacity = (2 - 7 + 10) mod 10 = 5.", ["array", "circular queue"], "hard"),
                self._q("In column-major storage of a 2D array with 1-based bounds A[1..4][1..5], which index changes fastest in memory?", ["The row index", "The column index", "Neither index", "Both indexes alternate randomly"], 0, "Column-major order stores a complete column contiguously, so the row index changes fastest.", ["array", "column-major"], "hard"),
                self._q("A dynamic array doubles capacity whenever full. What is the amortized insertion cost at the end?", ["O(1)", "O(log n)", "O(n)", "O(n log n)"], 0, "Occasional O(n) resizing is spread over many O(1) appends, giving amortized constant time.", ["dynamic array", "amortized"], "hard"),
                self._q("Which prefix-sum expression gives the sum of A[l..r] when P[i] stores A[0] through A[i-1]?", ["P[r + 1] - P[l]", "P[r] - P[l]", "P[l] - P[r + 1]", "P[r + 1] + P[l]"], 0, "With an exclusive prefix array, P[r+1] includes through r and P[l] excludes before l.", ["array", "prefix sum"], "hard"),
                self._q("What is the worst-case time to find the first occurrence of x in an unsorted array of n elements?", ["O(n)", "O(log n)", "O(1)", "O(n log n)"], 0, "Without ordering or indexing, each element may need to be checked once.", ["array", "linear search"], "easy"),
                self._q("Which representation is usually more space-efficient for a sparse matrix than a full 2D array?", ["Triples storing row, column, and value", "A second full zero matrix", "Only the diagonal length", "A stack of all zero entries"], 0, "Sparse matrix triples store only non-zero values and their coordinates.", ["array", "sparse matrix"], "medium"),
            ],
            "data": [
                self._q("Which data structure gives O(1) average-time lookup by key?", ["Hash table", "Stack", "Queue", "Sorted linked list"], 0, "A hash table maps keys to buckets, so lookup is O(1) on average when collisions are controlled.", ["hashing", "data structures"]),
                self._q("In a binary heap stored in an array, where is the parent of index i located for 0-based indexing?", ["floor((i - 1) / 2)", "2i + 1", "2i + 2", "i / 2 + 1"], 0, "For 0-based heap storage, children are 2i+1 and 2i+2, so the parent is floor((i-1)/2).", ["heap", "array"]),
                self._q("Which operation on a stack removes the most recently inserted element?", ["Pop", "Dequeue", "Peek without removal", "Enqueue"], 0, "A stack follows LIFO order, so pop removes the last element pushed.", ["stack", "lifo"]),
                self._q("Which tree traversal visits left subtree, root, then right subtree?", ["Inorder traversal", "Preorder traversal", "Postorder traversal", "Level-order traversal"], 0, "Inorder traversal is defined as left-root-right for binary trees.", ["tree", "traversal"]),
                self._q("Which collision-resolution method keeps all hash table entries inside the table array?", ["Open addressing", "Separate chaining", "Adjacency list", "External sorting"], 0, "Open addressing probes alternative slots in the same table instead of storing chains outside the array.", ["hashing", "collision"]),
                self._q("Consider a hash table with open addressing and linear probing. After deleting an item from the middle of a probe chain, which marker preserves future searches?", ["A tombstone marker", "A null empty slot", "A stack pointer", "A sorted duplicate key"], 0, "A tombstone keeps the probe chain searchable; replacing it with a plain empty slot can incorrectly stop lookup.", ["hashing", "open addressing"], "hard"),
                self._q("Given a min-heap array [4, 7, 6, 12, 9], which insertion of value 3 restores the heap property after bubbling up?", ["Place 3 at the end, then swap with 6 and 4", "Place 3 at the root without moving others", "Place 3 after 12 only", "Swap 3 only with 9"], 0, "Inserting 3 at the next leaf and bubbling it up through its parents restores the min-heap order.", ["heap", "array"], "hard"),
            ],
            "algorithm": [
                self._q("What is the worst-case time complexity of binary search on a sorted array of n elements?", ["O(log n)", "O(n)", "O(n log n)", "O(1)"], 0, "Binary search halves the search interval each step, giving O(log n) comparisons.", ["binary search", "complexity"]),
                self._q("Which technique is most suitable when a problem has overlapping subproblems and optimal substructure?", ["Dynamic programming", "Linear probing", "Round-robin scheduling", "Two-phase locking"], 0, "Dynamic programming stores subproblem results and combines optimal substructure efficiently.", ["dynamic programming", "algorithms"]),
                self._q("Which graph traversal uses a queue to visit vertices level by level?", ["Breadth-first search", "Depth-first search", "Kruskal algorithm", "Topological sort only"], 0, "BFS uses a queue, which naturally processes vertices in increasing distance from the start node.", ["graph", "bfs"]),
                self._q("Which sorting algorithm has O(n log n) average time and partitions around a pivot?", ["Quick sort", "Bubble sort", "Insertion sort", "Counting sort only"], 0, "Quick sort partitions around a pivot and has O(n log n) average time.", ["sorting", "quick sort"]),
                self._q("Dijkstra's algorithm requires which condition for edge weights?", ["Non-negative edge weights", "All weights must be equal", "Negative cycles must exist", "The graph must be a tree"], 0, "Dijkstra's greedy relaxation is correct when all edge weights are non-negative.", ["graph", "dijkstra"]),
                self._q("Which paradigm is used by Kruskal's minimum spanning tree algorithm?", ["Greedy method", "Backtracking only", "Divide and conquer only", "Dynamic programming only"], 0, "Kruskal repeatedly chooses the lightest safe edge, which is a greedy strategy.", ["greedy", "mst"]),
                self._q("Given edge weights with one negative edge but no negative cycle, which shortest-path algorithm remains appropriate?", ["Bellman-Ford", "Dijkstra", "Breadth-first search", "Kruskal"], 0, "Bellman-Ford handles negative edge weights, while Dijkstra's greedy choice may fail with a negative edge.", ["graph", "shortest path"], "hard"),
                self._q("If a recursive problem has overlapping subproblems but no optimal substructure, which statement is most accurate?", ["Memoization can avoid recomputation but may not produce an optimization recurrence", "Greedy choice is always correct", "Dynamic programming always applies unchanged", "Binary search must be used"], 0, "Overlapping subproblems help memoization, but optimization DP also needs optimal substructure.", ["dynamic programming", "recursion"], "hard"),
                self._q("Consider quick sort with an already sorted array and first-element pivot. What is the worst-case recurrence?", ["T(n) = T(n-1) + O(n)", "T(n) = 2T(n/2) + O(n)", "T(n) = T(n/2) + O(1)", "T(n) = O(1)"], 0, "A first-element pivot on sorted data creates one empty partition and one size n-1 partition, giving quadratic behavior.", ["sorting", "quick sort"], "hard"),
            ],
            "python": [
                self._q("In Python, which built-in data type is mutable and ordered?", ["list", "tuple", "str", "frozenset"], 0, "A Python list preserves insertion order and can be modified in place, so it is mutable.", ["python", "list", "mutable"]),
                self._q("In a Python dictionary, what happens when assigning a value to an existing key?", ["The old value is replaced", "A duplicate key is stored", "The key becomes immutable", "A syntax error is always raised"], 0, "Python dictionaries keep unique keys, so assigning the same key updates its associated value.", ["python", "dict", "dictionary"]),
                self._q("Which Python type is immutable and commonly used for fixed ordered records?", ["tuple", "list", "dict", "set"], 0, "A Python tuple is ordered but immutable, making it useful for fixed collections of values.", ["python", "tuple", "immutable"]),
                self._q("What does range(3) produce when iterated in Python?", ["0, 1, 2", "1, 2, 3", "0, 1, 2, 3", "3, 2, 1"], 0, "Python range(3) starts at zero by default and stops before the endpoint 3.", ["python", "range", "control flow"]),
                self._q("Which Python block handles an exception raised inside a try block?", ["except", "finally only", "class", "lambda"], 0, "In Python, an except block catches matching exceptions raised while the try block runs.", ["python", "exception"]),
                self._q("What is the main purpose of __init__ in a Python class?", ["Initialize object state when an instance is created", "Delete the class definition", "Import a module automatically", "Convert every object to a string"], 0, "__init__ is called during object construction and is commonly used to initialize instance attributes.", ["python", "class", "object"]),
                self._q("What is the result of the Python expression [x * x for x in range(3)]?", ["[0, 1, 4]", "[1, 4, 9]", "[0, 1, 2]", "[3, 6, 9]"], 0, "The list comprehension squares each Python range value 0, 1, and 2, producing 0, 1, and 4.", ["python", "comprehension", "list"]),
                self._q("Which Python keyword is used inside a function to create a generator?", ["yield", "return only", "import", "raise only"], 0, "A Python function containing yield returns a generator object that produces values lazily.", ["python", "generator", "iterator"]),
                self._q("In Python, what does a decorator most directly do?", ["Wrap or modify a function or class", "Declare a variable as private", "Sort a list in place", "Catch all exceptions globally"], 0, "A Python decorator is applied to a function or class to wrap, replace, or extend its behavior.", ["python", "decorator", "function"]),
                self._q("Why is with open(path) commonly preferred for Python file handling?", ["It closes the file automatically when the block exits", "It prevents every possible IOError", "It converts text to bytecode", "It disables exceptions in the block"], 0, "The Python with statement uses a context manager so the file is closed reliably after the block.", ["python", "file handling", "context manager"]),
                self._q("In Python's LEGB rule, which scope is checked first for a variable name inside a function?", ["Local scope", "Built-in scope", "Global scope", "Enclosing module cache"], 0, "Python resolves names inside a function by checking local scope before enclosing, global, and built-in scopes.", ["python", "scope", "function"]),
                self._q("What does a Python lambda expression create?", ["An anonymous function object", "A mutable dictionary", "A compiled module file", "A handled exception"], 0, "A Python lambda expression creates a small anonymous function object with a single expression body.", ["python", "lambda", "function"]),
                self._q("What is the difference between == and is in Python?", ["== compares values, while is checks object identity", "== checks type only, while is compares values", "Both always compare memory addresses", "is can only be used with strings"], 0, "In Python, == calls equality comparison for values, while is tests whether two references point to the same object.", ["python", "object", "identity"]),
                self._q("Consider this Python code:\n\nitems = []\nfor value in range(3):\n    items.append(lambda: value)\nprint([fn() for fn in items])\n\nWhat is printed?", ["[2, 2, 2]", "[0, 1, 2]", "[1, 2, 3]", "A NameError is raised"], 0, "Python closures capture the variable, not its value at each loop iteration, so all lambdas see the final value 2.", ["python", "closure", "lambda"], "hard"),
                self._q("Given this Python function:\n\ndef add(value, bucket=[]):\n    bucket.append(value)\n    return bucket\n\nWhat does add(1), then add(2) return on the second call?", ["[1, 2]", "[2]", "[1]", "A TypeError is raised"], 0, "Default list arguments are created once when the function is defined, so the same mutable list is reused.", ["python", "function", "mutable"], "hard"),
                self._q("Consider this Python expression:\n\nx = [1, 2]\ny = x\nx += [3]\nprint(y)\n\nWhat is printed?", ["[1, 2, 3]", "[1, 2]", "[3]", "A TypeError is raised"], 0, "The += operation mutates the original list object in place, so y sees the updated list.", ["python", "list", "identity"], "hard"),
                self._q("Trace this Python generator:\n\ndef gen():\n    yield from range(2)\n    return 5\n\nlist(gen())\n\nWhat is the resulting list?", ["[0, 1]", "[0, 1, 5]", "[5]", "A StopIteration is shown in the list"], 0, "The generator's return value becomes StopIteration metadata and is not included by list().", ["python", "generator", "yield"], "hard"),
                self._q("Consider this Python class hierarchy:\n\nclass A:\n    value = []\nclass B(A):\n    pass\nB.value.append(1)\nprint(A.value)\n\nWhat is printed?", ["[1]", "[]", "AttributeError", "[[], 1]"], 0, "B initially inherits the same class attribute list from A, so mutating it through B is visible on A.", ["python", "class", "mutable"], "hard"),
            ],
            "database": [
                self._q("Which normal form removes transitive dependency of non-prime attributes on a key?", ["Third normal form", "First normal form", "Second normal form", "Domain-key normal form"], 0, "3NF disallows transitive dependency of non-prime attributes on candidate keys.", ["normalization", "dbms"]),
                self._q("Which ACID property ensures committed transaction effects survive a crash?", ["Durability", "Isolation", "Consistency", "Atomicity"], 0, "Durability means committed changes are persisted even after system failure.", ["transaction", "acid"]),
                self._q("A B+ tree index is preferred in databases mainly because it supports what efficiently?", ["Range queries with balanced search", "Only LIFO access", "Lossy compression", "Deadlock detection"], 0, "B+ trees keep sorted keys in leaves and stay balanced, making point and range queries efficient.", ["indexing", "b+ tree"]),
                self._q("In SQL, which clause filters grouped rows after aggregation?", ["HAVING", "WHERE", "ORDER BY", "DISTINCT"], 0, "HAVING applies conditions to groups after GROUP BY aggregation.", ["sql", "group by"]),
                self._q("Which anomaly can normalization reduce in a poorly designed relation?", ["Update anomaly", "Page fault", "Cache miss", "Packet collision"], 0, "Normalization decomposes relations to reduce insert, delete, and update anomalies.", ["normalization", "anomaly"]),
                self._q("Which schedule property means the result is equivalent to some serial execution?", ["Serializability", "Fragmentation", "Index clustering", "Domain independence"], 0, "A serializable schedule preserves correctness by matching the effect of a serial transaction order.", ["transaction", "serializability"]),
                self._q("Which key uniquely identifies a tuple in a relation?", ["Candidate key", "Foreign attribute only", "Multivalued dependency", "Derived attribute"], 0, "A candidate key is a minimal set of attributes that uniquely identifies each tuple.", ["key", "relational model"]),
                self._q("Which SQL command is used to remove rows satisfying a condition?", ["DELETE", "DROP DATABASE", "ALTER TABLE", "CREATE VIEW"], 0, "DELETE removes selected rows from a table while preserving the table schema.", ["sql", "delete"]),
                self._q("Which concurrency-control protocol uses shared and exclusive locks?", ["Two-phase locking", "Hash join", "View maintenance", "ER specialization"], 0, "Two-phase locking coordinates transactions using shared and exclusive locks to ensure serializability.", ["locking", "concurrency"]),
                self._q("In an ER model, a weak entity is identified using what?", ["Owner entity key plus partial key", "Only a multivalued attribute", "A derived attribute", "A file pointer"], 0, "A weak entity depends on an owner entity and uses a partial key with the owner's key.", ["er model", "weak entity"]),
            ],
            "operating": [
                self._q("Which deadlock condition means a resource cannot be forcibly taken from a process?", ["No preemption", "Mutual exclusion", "Hold and wait", "Circular wait"], 0, "No preemption says resources are released only voluntarily, one of Coffman's deadlock conditions.", ["deadlock", "os"]),
                self._q("In paging, what does a page fault indicate?", ["The referenced page is not currently in main memory", "The CPU cache is full", "A process completed normally", "The disk has no file system"], 0, "A page fault occurs when the needed virtual page must be brought into RAM.", ["paging", "memory"]),
                self._q("Which synchronization primitive can be used to protect a critical section?", ["Mutex", "Spooler", "Loader", "Assembler"], 0, "A mutex allows only one thread or process to enter a critical section at a time.", ["synchronization", "mutex"]),
                self._q("Which CPU scheduling algorithm can cause starvation when lower-priority processes wait indefinitely?", ["Priority scheduling", "Round robin with fixed quantum", "FCFS only", "Shortest queue routing"], 0, "Priority scheduling may starve low-priority processes unless aging or a similar technique is used.", ["scheduling", "starvation", "os"], "medium"),
                self._q("Banker's algorithm is used primarily for what?", ["Deadlock avoidance", "Page replacement", "Disk formatting", "File compression"], 0, "Banker's algorithm checks safe states before allocation to avoid deadlock.", ["deadlock", "bankers algorithm"], "medium"),
                self._q("In a system with available vector (3, 3, 2), Banker's algorithm checks whether what exists?", ["A safe sequence", "A page table index only", "A socket connection", "A compiler token stream"], 0, "The algorithm determines whether processes can finish in some safe order with available resources.", ["deadlock", "safe sequence"], "hard"),
                self._q("Which page replacement policy removes the page that will not be used for the longest future time?", ["Optimal replacement", "FIFO", "Round robin", "Shortest seek time first"], 0, "Optimal page replacement is theoretical and replaces the page whose next use is farthest in the future.", ["paging", "replacement"], "medium"),
                self._q("If a process holds one resource and waits for another, which deadlock condition is involved?", ["Hold and wait", "No preemption only", "Mutual exclusion only", "Bounded waiting"], 0, "Hold and wait means a process keeps allocated resources while requesting more.", ["deadlock", "hold and wait"], "easy"),
                self._q("Which mechanism maps virtual page numbers to physical frame numbers?", ["Page table", "Ready queue", "Semaphore table", "File descriptor only"], 0, "The page table stores mappings from virtual pages to physical frames.", ["paging", "page table"], "medium"),
                self._q("Which disk scheduling algorithm services the closest pending request first?", ["SSTF", "FCFS", "Round robin", "LRU"], 0, "Shortest Seek Time First chooses the pending request requiring the least head movement.", ["disk scheduling", "sstf"], "medium"),
                self._q("Consider three processes with burst times 4, 6, and 10 ms under FCFS. What is the average waiting time if they arrive together in that order?", ["14/3 ms", "20/3 ms", "10 ms", "0 ms"], 0, "Waiting times are 0, 4, and 10 ms, so the average is (0 + 4 + 10) / 3 = 14/3 ms.", ["process", "scheduling", "os"], "hard"),
                self._q("Determine the number of page frames needed if a process has 16 KB logical address space and page size is 4 KB.", ["4 pages", "2 pages", "8 pages", "64 pages"], 0, "Number of pages equals address space divided by page size: 16 KB / 4 KB = 4.", ["paging", "process", "memory"], "medium"),
            ],
            "network": [
                self._q("Which transport protocol provides reliable, ordered byte-stream delivery?", ["TCP", "UDP", "IP", "ARP"], 0, "TCP adds sequencing, acknowledgements, and retransmission for reliable ordered delivery.", ["tcp", "transport"]),
                self._q("Which device primarily forwards packets using IP addresses?", ["Router", "Repeater", "Hub", "NIC"], 0, "Routers use network-layer IP addresses to choose next hops between networks.", ["routing", "ip"]),
                self._q("What does subnetting primarily help achieve?", ["Dividing an IP network into smaller logical networks", "Encrypting every packet", "Replacing TCP", "Increasing MAC address length"], 0, "Subnetting borrows host bits to create smaller networks and manage routing/address allocation.", ["subnet", "ip"]),
                self._q("In IPv4, a /24 network leaves how many host bits?", ["8", "24", "16", "32"], 0, "IPv4 has 32 bits, so /24 leaves 8 host bits.", ["subnet", "ipv4"], "medium"),
                self._q("Which OSI layer is responsible for end-to-end process communication using ports?", ["Transport layer", "Physical layer", "Data link layer", "Presentation layer only"], 0, "The transport layer uses ports and protocols such as TCP or UDP for process-to-process delivery.", ["transport", "osi", "ports"], "medium"),
                self._q("What does ARP resolve on an IPv4 LAN?", ["IP address to MAC address", "Domain name to IP address", "Port to process id", "Ciphertext to plaintext"], 0, "ARP maps an IPv4 address to a link-layer MAC address on the local network.", ["arp", "mac", "network"], "medium"),
                self._q("Which protocol is commonly used to translate domain names to IP addresses?", ["DNS", "SMTP", "SSH", "ICMP"], 0, "DNS resolves human-readable domain names into IP addresses.", ["dns", "application layer"], "easy"),
                self._q("If a subnet mask is 255.255.255.0, what is the prefix length?", ["/24", "/16", "/8", "/32"], 0, "Three 255 octets represent 24 network bits.", ["subnet", "mask"], "easy"),
                self._q("Which congestion-control behavior reduces TCP's congestion window after packet loss?", ["Multiplicative decrease", "Address resolution", "Static routing only", "Name caching"], 0, "TCP congestion control backs off the congestion window after detecting loss.", ["tcp", "congestion"], "hard"),
            ],
            "theory": [
                self._q("Which machine model recognizes exactly the regular languages?", ["Finite automaton", "Turing machine only", "Pushdown automaton only", "Linear bounded automaton only"], 0, "DFA and NFA finite automata are equivalent recognizers for regular languages.", ["dfa", "regular language"]),
                self._q("Which grammar class is accepted by a pushdown automaton?", ["Context-free grammar", "Regular expression only", "Unrestricted grammar only", "Attribute grammar only"], 0, "Pushdown automata use a stack and accept context-free languages.", ["pda", "cfg"]),
                self._q("In compiler design, lexical analysis mainly produces what?", ["Tokens", "Machine code", "Parse trees only", "Register allocation"], 0, "The lexer scans characters and groups them into tokens for the parser.", ["compiler", "lexer"]),
            ],
            "biology": [
                self._q("Which cell organelle is called the powerhouse of the cell?", ["Mitochondria", "Ribosome", "Nucleus only", "Cell wall"], 0, "Mitochondria release energy through cellular respiration and produce ATP.", ["biology", "cell", "mitochondria"], "easy"),
                self._q("Which pigment captures light energy during photosynthesis?", ["Chlorophyll", "Haemoglobin", "Keratin", "Insulin"], 0, "Chlorophyll in chloroplasts absorbs light energy for photosynthesis.", ["photosynthesis", "chlorophyll", "biology"], "easy"),
                self._q("What is the main function of xylem in plants?", ["Transporting water and minerals", "Transporting prepared food only", "Making hormones", "Producing gametes"], 0, "Xylem carries water and dissolved minerals from roots to other plant parts.", ["xylem", "transport", "biology"], "medium"),
                self._q("Which openings in leaves regulate gas exchange for photosynthesis?", ["Stomata", "Root hairs", "Sepals", "Anthers"], 0, "Stomata are pores that allow exchange of carbon dioxide, oxygen, and water vapor.", ["stomata", "photosynthesis", "biology"], "medium"),
                self._q("Which enzyme in saliva begins digestion of starch?", ["Amylase", "Pepsin", "Trypsin", "Lipase only"], 0, "Salivary amylase starts breaking starch into simpler sugars in the mouth.", ["enzyme", "digestion", "biology"], "easy"),
                self._q("In human blood, which cells mainly transport oxygen?", ["Red blood cells", "Platelets", "White blood cells", "Plasma proteins only"], 0, "Red blood cells contain haemoglobin, which binds and transports oxygen.", ["respiration", "blood", "biology"], "medium"),
                self._q("Why does the rate of photosynthesis fall when carbon dioxide becomes limiting?", ["Carbon dioxide is a raw material for glucose formation", "Carbon dioxide destroys chlorophyll", "Carbon dioxide blocks stomata permanently", "Carbon dioxide turns light into heat"], 0, "Photosynthesis uses carbon dioxide to build glucose, so a shortage limits the process.", ["photosynthesis", "carbon dioxide", "biology"], "hard"),
                self._q("Which hormone lowers blood glucose level in humans?", ["Insulin", "Adrenaline", "Thyroxine", "Growth hormone"], 0, "Insulin helps cells take up glucose and lowers blood glucose concentration.", ["hormone", "insulin", "biology"], "medium"),
                self._q("Which part of a flower produces pollen grains?", ["Anther", "Stigma", "Ovary", "Sepal"], 0, "The anther is part of the stamen and produces pollen grains.", ["reproduction", "flower", "biology"], "easy"),
                self._q("A plant is kept in darkness for two days before a starch test. What is the main purpose?", ["To destarch the leaves", "To kill chlorophyll", "To increase transpiration", "To close roots permanently"], 0, "Keeping a plant in darkness uses stored starch, so new starch formation can be tested.", ["photosynthesis", "starch", "biology"], "hard"),
                self._q("Which type of reproduction creates offspring genetically identical to the parent?", ["Asexual reproduction", "Sexual reproduction", "Cross-pollination only", "Fertilization only"], 0, "Asexual reproduction uses one parent and usually produces genetically identical offspring.", ["reproduction", "heredity", "biology"], "medium"),
            ],
            "genetics": [
                self._q("In a monohybrid cross of two heterozygous plants, what phenotypic ratio is expected under complete dominance?", ["3:1", "1:1", "9:3:3:1", "2:1"], 0, "Aa x Aa gives three dominant phenotype offspring for every one recessive phenotype.", ["genetics", "mendel"]),
                self._q("Which molecule carries genetic information from DNA to ribosomes during protein synthesis?", ["mRNA", "tRNA", "rRNA", "ATP"], 0, "mRNA is transcribed from DNA and carries codon information to the ribosome.", ["dna", "rna"]),
                self._q("A codon consists of how many nucleotides?", ["Three", "Two", "Four", "One"], 0, "Each codon is a triplet of nucleotides that specifies an amino acid or stop signal.", ["codon", "genetic code"]),
                self._q("Which enzyme synthesizes a new DNA strand during replication?", ["DNA polymerase", "RNA ligase", "Pepsin", "Amylase"], 0, "DNA polymerase adds nucleotides to form the new complementary DNA strand.", ["dna", "replication"]),
                self._q("Which term describes different forms of the same gene?", ["Alleles", "Codons", "Ribosomes", "Centromeres"], 0, "Alleles are alternate versions of a gene found at the same locus.", ["allele", "gene"]),
                self._q("In a dihybrid cross with independent assortment, what phenotypic ratio is expected in F2?", ["9:3:3:1", "3:1", "1:2:1", "1:1"], 0, "Independent assortment of two traits in AaBb x AaBb produces a 9:3:3:1 phenotypic ratio.", ["mendel", "dihybrid"]),
                self._q("Which nitrogenous base is present in RNA but not DNA?", ["Uracil", "Thymine", "Guanine", "Cytosine"], 0, "RNA uses uracil in place of thymine, while both DNA and RNA contain guanine and cytosine.", ["rna", "base"]),
                self._q("Crossing over occurs during which stage of meiosis?", ["Prophase I", "Metaphase II", "Anaphase II", "Telophase I"], 0, "Homologous chromosomes exchange segments during prophase I of meiosis.", ["meiosis", "crossing over"]),
                self._q("Which inheritance pattern shows both alleles fully expressed in a heterozygote?", ["Codominance", "Complete dominance", "Polyploidy", "Epistasis only"], 0, "In codominance, both alleles contribute visibly to the phenotype.", ["codominance", "inheritance"]),
                self._q("A test cross is usually performed with which genotype?", ["Homozygous recessive", "Homozygous dominant", "Heterozygous dominant only", "Polygenic dominant"], 0, "Crossing with a homozygous recessive individual reveals the unknown genotype.", ["test cross", "genetics"]),
            ],
            "physics": [
                self._q("Calculate the electrostatic force expression for charges +2 uC and +3 uC separated by 0.3 m.", ["k(2 x 10^-6)(3 x 10^-6)/(0.3)^2", "k(2 + 3)/(0.3)", "k(0.3)^2/(6 x 10^-12)", "k(2 x 10^-6)/(3 x 10^-6)"], 0, "Coulomb's law gives F = k q1 q2 / r^2 using charges in coulombs and distance in meters.", ["electrostatics", "force", "charge"], "hard"),
                self._q("Calculate the electric field magnitude when a 4 uC charge experiences force 0.2 N.", ["50000 N/C", "0.8 N/C", "5 x 10^-5 N/C", "200000 N/C"], 0, "Electric field E = F/q = 0.2 / (4 x 10^-6) = 50000 N/C.", ["electric field", "electrostatics", "charge"], "hard"),
                self._q("Calculate the work needed to move 2 C charge through an electric potential difference of 12 V.", ["24 J", "6 J", "14 J", "10 J"], 0, "Work equals qV, so 2 x 12 = 24 J.", ["potential", "work", "electrostatics"], "medium"),
                self._q("Calculate the charge stored by a 5 uF capacitor connected to 10 V.", ["50 uC", "2 uC", "15 uC", "0.5 uC"], 0, "Charge on a capacitor is Q = CV = 5 uF x 10 V = 50 uC.", ["capacitance", "charge", "electrostatics"], "medium"),
                self._q("Determine the equivalent capacitance of two identical 6 uF capacitors connected in series.", ["3 uF", "12 uF", "6 uF", "36 uF"], 0, "For two equal capacitors in series, equivalent capacitance is C/2 = 3 uF.", ["capacitance", "series", "electrostatics"], "hard"),
                self._q("Which statement is correct for electric field lines?", ["They start on positive charges and end on negative charges", "They always form closed loops in electrostatics", "They cross wherever field is strong", "They point from lower to higher potential only"], 0, "Electrostatic field lines originate on positive charge and terminate on negative charge or infinity.", ["electric field", "field lines", "electrostatics"], "medium"),
                self._q("Determine the fraction of electric field magnitude when distance from a point charge is doubled.", ["One-fourth", "One-half", "Double", "Four times"], 0, "For a point charge, E is proportional to 1/r^2, so doubling r reduces E to one-fourth.", ["electric field", "inverse square", "electrostatics"], "medium"),
                self._q("Calculate the equivalent capacitance of 2 uF and 4 uF capacitors connected in parallel.", ["6 uF", "2 uF", "4/3 uF", "8 uF"], 0, "Capacitances in parallel add directly, so 2 uF + 4 uF = 6 uF.", ["capacitance", "parallel", "electrostatics"], "medium"),
                self._q("Calculate the potential at a point 0.2 m from a +5 uC charge.", ["k(5 x 10^-6)/0.2", "k(0.2)/(5 x 10^-6)", "k(5 x 10^-6)(0.2)^2", "5 x 10^-6/k"], 0, "Potential due to a point charge is V = kq/r.", ["potential", "charge", "electrostatics"], "hard"),
                self._q("Determine the energy stored in a 4 uF capacitor charged to 100 V.", ["0.02 J", "0.2 J", "2 J", "400 J"], 0, "Energy is 1/2 CV^2 = 0.5 x 4 x 10^-6 x 10000 = 0.02 J.", ["capacitance", "energy", "electrostatics"], "hard"),
                self._q("Calculate the electric flux through a surface when E = 200 N/C, area = 0.5 m^2, and the field is normal to the surface.", ["100 N m^2/C", "400 N m^2/C", "0.0025 N m^2/C", "200.5 N m^2/C"], 0, "Flux is EA cos 0 = 200 x 0.5 = 100 N m^2/C.", ["electric field", "flux", "electrostatics"], "hard"),
                self._q("Determine the force on a 3 uC charge placed in a uniform electric field of 4000 N/C.", ["0.012 N", "12 N", "1.33 N", "0.0013 N"], 0, "Force is qE = 3 x 10^-6 x 4000 = 0.012 N.", ["electric field", "force", "electrostatics"], "medium"),
                self._q("If net force on a body is doubled while mass is constant, what happens to acceleration?", ["It doubles", "It halves", "It becomes zero", "It remains unchanged"], 0, "Newton's second law gives a = F/m, so acceleration is directly proportional to net force.", ["force", "acceleration"]),
                self._q("Ohm's law relates potential difference V, current I, and resistance R as which equation?", ["V = IR", "I = VR", "R = VI", "V = I/R"], 0, "Ohm's law states that voltage across a conductor equals current times resistance.", ["current", "resistance"]),
                self._q("For a convex lens, an object placed beyond 2F forms which kind of image?", ["Real, inverted, and diminished", "Virtual, erect, and enlarged", "Real, erect, and same size", "Virtual and diminished"], 0, "A convex lens forms a real, inverted, diminished image between F and 2F for objects beyond 2F.", ["optics", "lens"]),
                self._q("A circuit has a 6 V battery connected to a 3 ohm resistor. What current flows through the resistor?", ["2 A", "0.5 A", "9 A", "18 A"], 0, "Using Ohm's law I = V/R, the current is 6/3 = 2 A.", ["electricity", "current", "ohm"], "medium"),
                self._q("Two resistors of 2 ohm and 4 ohm are connected in series. What is their equivalent resistance?", ["6 ohm", "2 ohm", "8 ohm", "4/3 ohm"], 0, "Resistances in series add directly, so 2 + 4 = 6 ohm.", ["electricity", "resistance", "series"], "medium"),
                self._q("Two identical 6 ohm resistors are connected in parallel. What is the equivalent resistance?", ["3 ohm", "6 ohm", "12 ohm", "1 ohm"], 0, "For two equal resistors in parallel, equivalent resistance is half of one resistor, so 3 ohm.", ["electricity", "parallel", "resistance"], "medium"),
                self._q("An electric bulb draws 0.5 A from a 220 V supply. What is its power?", ["110 W", "220 W", "440 W", "0.002 W"], 0, "Electric power is P = VI, so 220 x 0.5 = 110 W.", ["electricity", "power", "current"], "medium"),
                self._q("Which device is used to measure electric current in a circuit?", ["Ammeter", "Voltmeter", "Galvanometer only for voltage", "Barometer"], 0, "An ammeter is connected in series to measure current in a circuit.", ["electricity", "current", "ammeter"], "easy"),
                self._q("A wire's resistance increases when which physical change is made?", ["Its length is increased", "Its length is decreased to zero", "Its area is increased", "It is replaced by a perfect conductor"], 0, "Resistance is directly proportional to length and inversely proportional to cross-sectional area.", ["electricity", "resistance", "wire"], "medium"),
                self._q("If three 3 ohm resistors are connected in parallel, what is the equivalent resistance?", ["1 ohm", "3 ohm", "6 ohm", "9 ohm"], 0, "For three equal resistors R in parallel, equivalent resistance is R/3, so 3/3 = 1 ohm.", ["electricity", "parallel", "resistance"], "hard"),
                self._q("A 60 W bulb runs for 5 hours. How much electrical energy does it consume?", ["0.30 kWh", "300 kWh", "12 kWh", "65 kWh"], 0, "Energy in kWh is power in kW multiplied by time: 0.060 x 5 = 0.30 kWh.", ["electricity", "energy", "power"], "hard"),
                self._q("A charge of 30 C passes through a conductor in 5 s. What is the current?", ["6 A", "150 A", "25 A", "0.17 A"], 0, "Current is charge per unit time, so I = Q/t = 30/5 = 6 A.", ["electricity", "charge", "current"], "medium"),
                self._q("Why is a voltmeter connected in parallel across a component?", ["It measures potential difference across the component", "It must carry the full circuit current", "It lowers the battery voltage to zero", "It replaces the component"], 0, "Voltage is measured between two points, so a voltmeter is placed in parallel across the component.", ["electricity", "voltage", "circuit"], "easy"),
                self._q("A ray of light going from air to glass bends toward the normal mainly because what changes?", ["Its speed decreases", "Its frequency becomes zero", "Its mass increases", "Its wavelength becomes infinite"], 0, "Light slows down in a denser medium like glass, causing refraction toward the normal.", ["light", "refraction", "physics"], "medium"),
                self._q("A 2 kg object moving at 3 m/s has kinetic energy equal to what?", ["9 J", "6 J", "3 J", "18 J"], 0, "Kinetic energy is 1/2 mv^2 = 1/2 x 2 x 3^2 = 9 J.", ["energy", "motion", "physics"], "medium"),
            ],
            "chemistry": [
                self._q("What is the number of particles in one mole of a substance?", ["6.022 x 10^23", "3.14 x 10^8", "9.8", "1.6 x 10^-19"], 0, "Avogadro's constant gives 6.022 x 10^23 particles per mole.", ["mole", "avogadro"]),
                self._q("Which bond generally forms by sharing electron pairs?", ["Covalent bond", "Ionic bond", "Metallic bond only", "Hydrogen bond only"], 0, "A covalent bond forms when atoms share electron pairs to complete valence shells.", ["bond", "covalent"]),
                self._q("For an exothermic reaction, what is the sign of enthalpy change?", ["Negative", "Positive", "Zero always", "Undefined"], 0, "Exothermic reactions release heat, so products have lower enthalpy and delta H is negative.", ["enthalpy", "thermodynamics"]),
                self._q("Which gas is usually released when an acid reacts with a metal carbonate?", ["Carbon dioxide", "Hydrogen", "Oxygen", "Nitrogen"], 0, "Acids react with metal carbonates to form salt, water, and carbon dioxide.", ["acid", "carbonate", "chemistry"], "easy"),
                self._q("What is the pH nature of a solution with pH 3?", ["Acidic", "Basic", "Neutral", "Saturated"], 0, "Solutions with pH below 7 are acidic.", ["acid", "base", "ph"], "easy"),
                self._q("In a balanced chemical equation, what must be equal on both sides?", ["Number of atoms of each element", "Number of words", "Only number of molecules", "Only state symbols"], 0, "The law of conservation of mass requires equal atoms of every element on both sides.", ["chemical reaction", "balancing"], "medium"),
                self._q("Which process coats iron with zinc to prevent rusting?", ["Galvanization", "Distillation", "Sublimation", "Crystallization"], 0, "Galvanization protects iron by coating it with zinc.", ["metal", "corrosion", "chemistry"], "medium"),
                self._q("Which carbon compound functional group is present in ethanol?", ["Alcohol group", "Carboxylic acid group", "Aldehyde group", "Ketone group"], 0, "Ethanol contains the hydroxyl alcohol functional group.", ["carbon", "alcohol", "chemistry"], "medium"),
                self._q("During oxidation of copper powder in air, what compound is formed?", ["Copper oxide", "Copper sulphate", "Copper carbonate", "Copper chloride"], 0, "Copper reacts with oxygen on heating to form black copper oxide.", ["oxidation", "copper", "chemistry"], "medium"),
                self._q("A neutralization reaction between HCl and NaOH forms which products?", ["NaCl and water", "Hydrogen and oxygen", "Carbon dioxide and salt", "Sodium and chlorine gas"], 0, "An acid and base neutralize to form a salt and water; here the salt is NaCl.", ["acid", "base", "salt"], "medium"),
                self._q("Which statement best explains why ionic compounds conduct electricity when molten?", ["Their ions can move freely", "Their molecules become neutral", "Their electrons disappear", "Their crystals become lighter"], 0, "Molten ionic compounds conduct because mobile ions carry charge.", ["ionic", "electricity", "chemistry"], "hard"),
            ],
            "math": [
                self._q("If f(x)=x^2, what is f'(x)?", ["2x", "x", "x^3", "2"], 0, "Using the power rule, d(x^2)/dx = 2x.", ["derivative", "calculus"]),
                self._q("For two independent events A and B, P(A and B) equals what?", ["P(A)P(B)", "P(A)+P(B)", "P(A)-P(B)", "P(A)/P(B)"], 0, "Independence means the joint probability factors as P(A) times P(B).", ["probability", "independent events"]),
                self._q("The roots of x^2 - 5x + 6 = 0 are what?", ["2 and 3", "1 and 6", "-2 and -3", "0 and 5"], 0, "Factoring gives (x-2)(x-3)=0, so the roots are 2 and 3.", ["quadratic", "algebra"]),
            ],
            "aptitude": [
                self._q("A price increases by 20% and then decreases by 20%. What is the net change?", ["4% decrease", "No change", "4% increase", "20% decrease"], 0, "Starting at 100, it becomes 120 then 96, which is a 4% decrease.", ["percentage", "aptitude"]),
                self._q("If A can finish work in 10 days and B in 15 days, how many days together?", ["6 days", "5 days", "12 days", "25 days"], 0, "Combined rate is 1/10 + 1/15 = 1/6, so they finish in 6 days.", ["time and work", "rate"]),
                self._q("In syllogism, if all A are B and all B are C, which conclusion follows?", ["All A are C", "All C are A", "No A are C", "Some B are not C"], 0, "The inclusion chain A subset B subset C implies all A are C.", ["syllogism", "reasoning"]),
            ],
            "teaching": [
                self._q("Which teaching method is most suitable for developing higher-order thinking in learners?", ["Problem-solving discussion", "Only dictating notes", "Rote repetition without feedback", "Reading slides silently"], 0, "Problem-solving discussion asks learners to analyze and apply ideas, which develops higher-order thinking.", ["teaching", "learner"]),
                self._q("In Bloom's taxonomy, which level involves judging the value of an argument or solution?", ["Evaluation", "Remembering", "Imitation", "Receiving"], 0, "Evaluation requires making judgments using criteria, so it is above simple recall.", ["teaching", "evaluation"]),
                self._q("Which assessment type is mainly used during instruction to improve learning?", ["Formative assessment", "Placement assessment only", "Terminal certification only", "Norming of institutions"], 0, "Formative assessment gives feedback during learning so teaching can be adjusted.", ["teaching", "assessment"]),
            ],
            "research": [
                self._q("A testable statement predicting a relationship between variables is called what?", ["Hypothesis", "Bibliography", "Appendix", "Index"], 0, "A hypothesis is a tentative, testable proposition about variables.", ["research", "hypothesis"]),
                self._q("Which sampling method gives every member of the population an equal chance of selection?", ["Simple random sampling", "Convenience sampling", "Snowball sampling", "Purposive sampling"], 0, "Simple random sampling selects units so each member has an equal probability of inclusion.", ["research", "sampling"]),
                self._q("Which research design is most appropriate for establishing cause-effect under controlled conditions?", ["Experimental design", "Historical narration", "Case report only", "Bibliographic listing"], 0, "Experimental designs manipulate variables under control to test causal effects.", ["research", "experimental"]),
            ],
            "communication": [
                self._q("Noise in communication primarily affects which part of the process?", ["Accurate transmission and decoding of message", "Only the sender's name", "Only the seating plan", "The legal status of the channel"], 0, "Noise interferes with transmission or decoding, reducing message accuracy.", ["communication", "barrier"]),
                self._q("Which communication pattern allows immediate feedback between teacher and learner?", ["Two-way communication", "One-way broadcast", "Anonymous notice only", "Static poster only"], 0, "Two-way communication allows both message delivery and response feedback.", ["communication", "feedback"]),
                self._q("Semantic barriers in communication are mainly related to what?", ["Meaning of words and symbols", "Room temperature only", "Network cable length only", "Font size only"], 0, "Semantic barriers arise when words, symbols, or meanings are interpreted differently.", ["communication", "semantic barrier"]),
            ],
            "logical": [
                self._q("If all researchers are scholars and some scholars are teachers, which conclusion definitely follows?", ["Some scholars are researchers", "All teachers are researchers", "No researcher is a teacher", "All scholars are teachers"], 0, "All researchers being scholars guarantees that at least those researchers are scholars.", ["logical reasoning", "syllogism"]),
                self._q("In an argument, a fallacy is best described as what?", ["An error in reasoning", "A verified conclusion", "A random sample", "A research tool"], 0, "A fallacy is a flaw in reasoning that weakens an argument.", ["logical reasoning", "fallacy"]),
                self._q("Which relation is represented by the sequence 2, 4, 8, 16?", ["Each term is multiplied by 2", "Each term increases by 1", "Each term is squared from previous", "Each term is divided by 2"], 0, "The sequence doubles each time, so every term is the previous term multiplied by 2.", ["logical reasoning", "series"]),
            ],
            "mathematical": [
                self._q("If 40% of a number is 80, what is the number?", ["200", "120", "160", "320"], 0, "If 0.40x = 80, then x = 80 / 0.40 = 200.", ["mathematical reasoning", "percentage"]),
                self._q("The average of 6, 8, 10, and 12 is what?", ["9", "8", "10", "11"], 0, "The sum is 36 and there are 4 values, so the average is 9.", ["mathematical reasoning", "average"]),
                self._q("If A:B = 2:3 and B:C = 6:5, what is A:C?", ["4:5", "2:5", "3:5", "5:4"], 0, "Make B common: A:B is 4:6 and B:C is 6:5, so A:C is 4:5.", ["mathematical reasoning", "ratio"]),
            ],
            "interpretation": [
                self._q("In a bar chart, the tallest bar usually represents what?", ["The category with the highest value", "The first category only", "The average of all categories", "The missing data point"], 0, "Bar height encodes value, so the tallest bar indicates the highest value.", ["data interpretation", "bar chart"]),
                self._q("If total students are 500 and 30% are in science, how many students are in science?", ["150", "130", "170", "200"], 0, "30% of 500 is 0.30 x 500 = 150.", ["data interpretation", "percentage"]),
                self._q("A pie chart sector angle of 90 degrees represents what fraction of the whole?", ["One-fourth", "One-half", "One-third", "Three-fourths"], 0, "90 degrees out of 360 degrees is 1/4 of the whole.", ["data interpretation", "pie chart"]),
            ],
            "ict": [
                self._q("Which ICT term refers to malicious software designed to harm or exploit systems?", ["Malware", "Firewall", "Spreadsheet", "Router table"], 0, "Malware is software intentionally designed to disrupt, damage, or gain unauthorized access.", ["ict", "malware"]),
                self._q("Which protocol is commonly used for secure web browsing?", ["HTTPS", "FTP only", "SMTP", "POP3"], 0, "HTTPS uses encryption over HTTP to secure web communication.", ["ict", "https"]),
                self._q("In digital learning, an LMS is mainly used for what?", ["Managing course content and learner activities", "Increasing screen brightness", "Replacing all assessment", "Compressing images only"], 0, "A learning management system organizes courses, materials, users, assessments, and tracking.", ["ict", "lms"]),
            ],
            "environment": [
                self._q("Sustainable development primarily tries to balance development with what?", ["Environmental protection and future needs", "Only industrial output", "Only urban expansion", "Ignoring resource limits"], 0, "Sustainable development meets present needs while preserving resources for future generations.", ["environment", "sustainable"]),
                self._q("Which gas is most associated with enhanced greenhouse effect from human activity?", ["Carbon dioxide", "Oxygen", "Nitrogen", "Argon"], 0, "Carbon dioxide from fossil fuels is a major anthropogenic greenhouse gas.", ["environment", "greenhouse"]),
                self._q("The concept of biodiversity refers to what?", ["Variety of living organisms at genetic, species, and ecosystem levels", "Only number of buildings", "Only rainfall amount", "Only mineral deposits"], 0, "Biodiversity includes variation within species, between species, and across ecosystems.", ["environment", "biodiversity"]),
            ],
            "higher": [
                self._q("Which body is primarily associated with maintaining standards in Indian university education?", ["University Grants Commission", "Election Commission", "Finance Commission", "Bar Council only"], 0, "The UGC coordinates and maintains standards of university education in India.", ["higher education", "ugc"]),
                self._q("NAAC accreditation is mainly related to what?", ["Quality assessment of higher education institutions", "Income tax collection", "Railway recruitment", "Weather forecasting"], 0, "NAAC assesses and accredits higher education institutions on quality parameters.", ["higher education", "naac"]),
                self._q("The National Education Policy 2020 emphasizes which higher-education reform?", ["Multidisciplinary and flexible education", "Only rote memorization", "Single-subject isolation only", "Removal of all assessment"], 0, "NEP 2020 emphasizes flexibility, multidisciplinarity, and holistic higher education.", ["higher education", "nep 2020"]),
            ],
            "polity": [
                self._q("Which part of the Indian Constitution contains Fundamental Rights?", ["Part III", "Part IV", "Part II", "Part IX"], 0, "Fundamental Rights are listed in Part III of the Indian Constitution.", ["constitution", "fundamental rights"]),
                self._q("Who is the constitutional head of a State in India?", ["Governor", "Chief Minister", "Speaker", "Advocate General"], 0, "The Governor is the constitutional head of the State executive.", ["governor", "polity"]),
                self._q("Which body is responsible for conducting elections to Parliament in India?", ["Election Commission of India", "Finance Commission", "NITI Aayog", "Lok Sabha Secretariat"], 0, "The Election Commission of India conducts and supervises parliamentary elections.", ["election", "constitution"]),
                self._q("Article 19 of the Indian Constitution primarily protects which category of rights?", ["Freedom rights", "Religious tax exemptions only", "Directive Principles", "Emergency powers"], 0, "Article 19 protects freedoms such as speech, assembly, association, movement, residence, and profession.", ["article 19", "fundamental rights"], "medium"),
                self._q("Which writ is issued to produce a detained person before a court?", ["Habeas corpus", "Mandamus", "Certiorari", "Quo warranto"], 0, "Habeas corpus protects personal liberty by requiring production of a detained person before court.", ["writ", "rights", "judiciary"], "medium"),
                self._q("Directive Principles of State Policy are mainly contained in which part?", ["Part IV", "Part III", "Part V", "Part IXA"], 0, "DPSPs are non-justiciable governance principles listed in Part IV.", ["directive principles", "constitution"], "easy"),
                self._q("Which doctrine means Parliament cannot destroy the essential features of the Constitution?", ["Basic structure doctrine", "Collective responsibility", "Double jeopardy", "Lame duck doctrine"], 0, "The basic structure doctrine limits constitutional amendments that damage essential constitutional features.", ["basic structure", "constitution"], "medium"),
                self._q("Which article is directly associated with abolition of untouchability?", ["Article 17", "Article 14", "Article 21", "Article 32"], 0, "Article 17 abolishes untouchability and forbids its practice in any form.", ["article 17", "fundamental rights"], "easy"),
                self._q("Consider a law that unreasonably restricts peaceful assembly. Which Fundamental Right is most directly affected?", ["Freedom under Article 19", "Right against exploitation only", "Cultural rights only", "Directive Principles only"], 0, "Peaceful assembly is part of the freedoms protected under Article 19, subject to reasonable restrictions.", ["article 19", "assembly", "rights"], "hard"),
                self._q("Which institution has the power of judicial review over laws violating Fundamental Rights?", ["Supreme Court and High Courts", "Election Commission only", "Finance Commission only", "Comptroller and Auditor General only"], 0, "The Supreme Court and High Courts can review laws and enforce Fundamental Rights through writ jurisdiction.", ["judicial review", "rights"], "medium"),
                self._q("Fundamental Duties were added to the Constitution by which amendment?", ["42nd Amendment", "44th Amendment", "1st Amendment", "73rd Amendment"], 0, "The 42nd Amendment inserted Fundamental Duties into Part IVA.", ["fundamental duties", "amendment"], "medium"),
            ],
            "history": [
                self._q("The Non-Cooperation Movement was launched after which major event?", ["Jallianwala Bagh massacre and Khilafat issue", "Partition of Bengal only", "Quit India resolution", "Dandi March"], 0, "The movement followed anger over Jallianwala Bagh and the Khilafat question.", ["modern history", "non cooperation"]),
                self._q("Who founded the Indian National Congress in 1885?", ["A. O. Hume", "M. G. Ranade", "Dadabhai Naoroji", "Gopal Krishna Gokhale"], 0, "A. O. Hume helped found the Indian National Congress in 1885.", ["history", "congress"]),
                self._q("The Dandi March was directly associated with protest against which tax?", ["Salt tax", "Land revenue only", "Income tax", "Textile tax"], 0, "Gandhi's Dandi March challenged the British monopoly and tax on salt.", ["history", "dandi march", "salt"], "easy"),
                self._q("Which 1905 event intensified the Swadeshi Movement in Bengal?", ["Partition of Bengal", "Champaran Satyagraha", "Cripps Mission", "Cabinet Mission"], 0, "The 1905 Partition of Bengal sparked boycott and Swadeshi protests.", ["history", "swadeshi", "bengal"], "medium"),
                self._q("The Quit India Movement was launched in which year?", ["1942", "1930", "1919", "1947"], 0, "The All India Congress Committee launched Quit India in August 1942.", ["history", "quit india"], "easy"),
                self._q("Which act is associated with the Rowlatt Satyagraha?", ["Rowlatt Act", "Government of India Act 1935", "Indian Councils Act 1861", "Pitt's India Act"], 0, "The Rowlatt Act allowed detention without trial and triggered nationwide protest.", ["history", "rowlatt act"], "medium"),
                self._q("Who led the Revolt of 1857 at Jhansi?", ["Rani Lakshmibai", "Begum Hazrat Mahal", "Sarojini Naidu", "Annie Besant"], 0, "Rani Lakshmibai became the prominent leader of resistance at Jhansi.", ["history", "1857", "jhansi"], "easy"),
                self._q("Which organization was formed by Bal Gangadhar Tilak and Annie Besant to demand self-government?", ["Home Rule League", "Ghadar Party", "Servants of India Society", "Forward Bloc"], 0, "Tilak and Annie Besant led Home Rule Leagues to campaign for self-government.", ["history", "home rule"], "medium"),
                self._q("The Poona Pact of 1932 was mainly between Gandhi and which leader?", ["B. R. Ambedkar", "Subhas Chandra Bose", "Jawaharlal Nehru", "Sardar Patel"], 0, "The Poona Pact was an agreement between Gandhi and Ambedkar on reserved seats.", ["history", "poona pact"], "medium"),
                self._q("Which session of Congress declared Poorna Swaraj as the goal?", ["Lahore Session 1929", "Surat Session 1907", "Calcutta Session 1886", "Tripuri Session 1939"], 0, "The Lahore Session of 1929 adopted Poorna Swaraj as the objective.", ["history", "poorna swaraj"], "medium"),
                self._q("Why was the Chauri Chaura incident significant for the Non-Cooperation Movement?", ["It led Gandhi to withdraw the movement", "It started the Salt March", "It created the Indian National Congress", "It ended the Rowlatt Act"], 0, "After violence at Chauri Chaura, Gandhi withdrew the Non-Cooperation Movement.", ["history", "chauri chaura"], "hard"),
                self._q("Which statement best distinguishes Civil Disobedience from Non-Cooperation?", ["Civil Disobedience openly broke specific laws like salt laws", "Civil Disobedience avoided all public protest", "Non-Cooperation began after the Dandi March", "Both were only armed movements"], 0, "Civil Disobedience involved deliberate violation of unjust laws, most famously the salt law.", ["history", "civil disobedience"], "hard"),
            ],
            "geography": [
                self._q("The southwest monsoon in India is primarily caused by what?", ["Seasonal pressure difference between land and sea", "Earthquake activity", "Ocean salinity only", "Tidal friction"], 0, "Differential heating creates pressure gradients that drive moisture-laden monsoon winds.", ["monsoon", "geography"]),
                self._q("Which soil is most associated with cotton cultivation in India?", ["Black soil", "Laterite soil", "Desert soil", "Alluvial soil only"], 0, "Black regur soil retains moisture and is well suited for cotton cultivation.", ["soil", "cotton"]),
            ],
            "science": [
                self._q("Which organelle is known as the powerhouse of the cell?", ["Mitochondria", "Ribosome", "Golgi body", "Lysosome"], 0, "Mitochondria produce ATP through cellular respiration.", ["cell", "mitochondria"]),
                self._q("Which process converts glucose into energy in cells?", ["Respiration", "Photosynthesis", "Osmosis", "Transpiration"], 0, "Cellular respiration breaks down glucose to release usable energy.", ["respiration", "life processes"]),
                self._q("Which form of energy is stored in a stretched rubber band?", ["Elastic potential energy", "Chemical energy", "Nuclear energy", "Sound energy"], 0, "A stretched rubber band stores energy due to deformation, called elastic potential energy.", ["energy", "physics", "science"], "easy"),
                self._q("Which separation method is best for separating sand from water?", ["Filtration", "Evaporation only", "Sublimation", "Chromatography only"], 0, "Filtration separates an insoluble solid such as sand from a liquid.", ["matter", "separation", "science"], "easy"),
                self._q("Which gas is needed by plants for photosynthesis?", ["Carbon dioxide", "Nitrogen", "Hydrogen", "Helium"], 0, "Plants use carbon dioxide and water to make glucose during photosynthesis.", ["photosynthesis", "biology", "science"], "easy"),
                self._q("What happens to current if voltage is doubled while resistance stays constant?", ["Current doubles", "Current halves", "Current becomes zero", "Current stays unchanged"], 0, "Ohm's law gives I = V/R, so doubling voltage doubles current when resistance is constant.", ["electricity", "current", "science"], "medium"),
                self._q("Which acid is naturally present in lemon juice?", ["Citric acid", "Hydrochloric acid", "Sulfuric acid", "Nitric acid"], 0, "Lemon juice contains citric acid, giving it a sour taste.", ["acid", "chemistry", "science"], "easy"),
                self._q("Why does a pencil partly immersed in water appear bent?", ["Refraction of light", "Reflection only", "Evaporation", "Magnetism"], 0, "Light changes direction when it passes between water and air, causing apparent bending.", ["light", "refraction", "science"], "medium"),
                self._q("Which blood component helps in clotting after injury?", ["Platelets", "Red blood cells", "Plasma water only", "Haemoglobin"], 0, "Platelets help form clots and stop bleeding at injury sites.", ["blood", "biology", "science"], "medium"),
                self._q("A metal spoon feels colder than a wooden spoon at the same room temperature because metal does what faster?", ["Conducts heat away from the hand", "Produces cold energy", "Absorbs oxygen", "Stops conduction"], 0, "Metal is a better conductor and removes heat from the hand more quickly.", ["heat", "conduction", "science"], "medium"),
            ],
        }

    def _q(self, prompt: str, options: list[str], correct_option_index: int, explanation: str, tags: list[str], difficulty: MockTestDifficulty | str = "auto") -> dict[str, Any]:
        return {
            "prompt": prompt,
            "options": options,
            "correct_option_index": correct_option_index,
            "explanation": explanation,
            "tags": tags,
            "difficulty": difficulty,
        }
