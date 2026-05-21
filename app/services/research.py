from __future__ import annotations

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urlparse, urlunparse

import httpx

from app.config import Settings
from app.models import AgentEvent, ResearchJobResponse, ResearchRequest, ResearchResponse, Source
from app.services.llm import LlmService
from app.services.reports import ReportService
from app.services.search import SearchService


@dataclass
class SourceDocument:
    source: Source
    text: str


@dataclass
class EvidenceCard:
    source_index: int
    claim: str
    support: str
    relevance: float = 0.5


class ReportQualityValidator:
    required_sections = [
        "Executive Summary",
        "Background",
        "Key Findings",
        "Detailed Analysis",
        "Tactical Playbook",
        "Latest Developments",
        "Limitations and Risks",
        "Conclusion",
        "References",
    ]
    mojibake_markers = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u0080", "\u00e2\u20ac\u2122", "\u00e2\u20ac\u0153")

    def validate(self, markdown: str, source_count: int) -> list[str]:
        report = markdown.strip()
        issues: list[str] = []
        if len(report) < 500:
            issues.append("report is too short")
        if self.has_mojibake(report):
            issues.append("report contains broken text encoding")
        if self.has_fake_metadata(report):
            issues.append("report contains fake memo metadata")
        if re.search(r"^## Detailed Report\s*\n\s*#", report, flags=re.IGNORECASE | re.MULTILINE):
            issues.append("report contains legacy wrapper markdown")
        if len(re.findall(r"^## References\s*$", report, flags=re.IGNORECASE | re.MULTILINE)) != 1:
            issues.append("report must contain exactly one References section")
        if self.has_dangling_text(report):
            issues.append("report appears truncated or has dangling punctuation")

        sections = self.sections(report)
        for section in self.required_sections:
            body = sections.get(section.lower(), "").strip()
            if not body:
                issues.append(f"missing required section: {section}")
                continue
            minimum = 20 if section == "References" else 45
            if len(re.sub(r"\s+", " ", body)) < minimum:
                issues.append(f"section is too thin: {section}")

        citation_ids = [int(match) for match in re.findall(r"\[(\d{1,2})\]", self.without_references(report))]
        invalid_ids = [citation_id for citation_id in citation_ids if citation_id < 1 or citation_id > source_count]
        if invalid_ids:
            issues.append("report contains invalid citation ids")
        if source_count and not citation_ids:
            issues.append("report has no inline citations")

        references = sections.get("references", "")
        if source_count and ("http://" not in references and "https://" not in references):
            issues.append("references do not contain source URLs")
        if citation_ids:
            ref_numbers = {int(match) for match in re.findall(r"^\s*(\d+)\.", references, flags=re.MULTILINE)}
            missing_refs = sorted(set(citation_ids) - ref_numbers)
            if missing_refs:
                issues.append("references do not cover cited sources")

        if self.has_overstated_pending_law(report):
            issues.append("report overstates pending law or bill status")
        return issues

    def sections(self, markdown: str) -> dict[str, str]:
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            name = match.group(1).strip().lower()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
            sections[name] = markdown[start:end].strip()
        return sections

    def without_references(self, markdown: str) -> str:
        return re.split(r"^## References\s*$", markdown, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]

    def has_mojibake(self, text: str) -> bool:
        return any(marker in text for marker in self.mojibake_markers)

    def has_fake_metadata(self, text: str) -> bool:
        return bool(re.search(r"^\s*\*{0,2}(date|to|from|subject)\*{0,2}\s*:", text, flags=re.IGNORECASE | re.MULTILINE))

    def has_dangling_text(self, text: str) -> bool:
        stripped = text.strip()
        if stripped.endswith(("[", "(", "{", ":", "-", "##")):
            return True
        if re.search(r"(\[[0-9,\s]*|\([^)]*)$", stripped):
            return True
        for section, body in self.sections(stripped).items():
            if section == "references":
                continue
            if body.strip().endswith(("[", "(", "{", ":", "-")):
                return True
        return False

    def has_overstated_pending_law(self, text: str) -> bool:
        law_context = r"(bill|act|h\.r\.|hr\s*\d+|legislation|regulation|law)"
        pending_context = r"(introduced|proposed|pending|referred|would|would require|if enacted)"
        enacted_claim = r"(now in effect|is in effect|has taken effect|requires developers|mandates developers|is now law)"
        return bool(
            re.search(law_context, text, flags=re.IGNORECASE)
            and re.search(enacted_claim, text, flags=re.IGNORECASE)
            and re.search(pending_context, text, flags=re.IGNORECASE)
        )


class _ReadableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "form"}
        self._break_tags = {"p", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag in self._break_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags and self._skip_depth:
            self._skip_depth -= 1
        if tag in self._break_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        clean = re.sub(r"\s+", " ", data).strip()
        if clean:
            self.parts.append(clean)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self.parts))


class ResearchService:
    def __init__(self, settings: Settings, llm: LlmService, search: SearchService, reports: ReportService):
        self.settings = settings
        self.llm = llm
        self.search = search
        self.reports = reports
        self.validator = ReportQualityValidator()
        self.jobs: dict[str, ResearchJobResponse] = {}

    def start_job(self, request: ResearchRequest) -> ResearchJobResponse:
        job = self._new_job(request)
        asyncio.create_task(self._run_job(job.id, save_report=True))
        return job

    async def run_to_completion(self, request: ResearchRequest, save_report: bool = False) -> ResearchJobResponse:
        job = self._new_job(request)
        await self._run_job(job.id, save_report=save_report)
        return job

    def get_job(self, job_id: str) -> ResearchJobResponse | None:
        return self.jobs.get(job_id)

    async def stream_events(self, job_id: str):
        sent = 0
        while True:
            job = self.jobs.get(job_id)
            if not job:
                yield 'event: error\ndata: {"error":"Research job not found."}\n\n'
                return

            while sent < len(job.events):
                yield f"data: {job.events[sent].model_dump_json()}\n\n"
                sent += 1

            if job.status in {"complete", "error"}:
                payload = {"done": True, "job_id": job.id, "status": job.status}
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                return

            await asyncio.sleep(0.7)

    def to_research_response(self, job: ResearchJobResponse) -> ResearchResponse:
        return ResearchResponse(
            summary=job.summary,
            detailed_answer=job.detailed_answer,
            citations=job.citations,
            confidence=job.confidence,
            critic_notes=job.critic_notes,
            events=job.events,
            setup_required=job.setup_required,
            job_id=job.id,
            status=job.status,
            report=job.report,
        )

    def _new_job(self, request: ResearchRequest) -> ResearchJobResponse:
        safe_topic = re.sub(r"[^a-z0-9]+", "-", request.topic.lower()).strip("-")[:36] or "research"
        job_id = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{safe_topic}-{uuid.uuid4().hex[:8]}"
        job = ResearchJobResponse(id=job_id, request=request)
        self.jobs[job_id] = job
        self._emit(job, "Query", "working", f"Research request queued: {request.topic}")
        return job

    async def _run_job(self, job_id: str, save_report: bool) -> None:
        job = self.jobs[job_id]
        request = job.request
        setup_required: list[str] = []

        try:
            self._set_status(job, "planning")
            queries = self._plan_queries(request)
            self._emit(job, "Query", "complete", f"Topic captured: {request.topic}")
            self._emit(job, "Supervisor", "complete", f"Planned {len(queries)} latest-web-first search queries.")

            self._set_status(job, "searching")
            self._emit(job, "Search Agents", "working", "Searching web and supporting academic providers.")
            candidates, search_setup = await self._discover_sources(request, queries)
            setup_required.extend(search_setup)
            self._emit(
                job,
                "Search Agents",
                "complete" if candidates else "warning",
                f"Collected {len(candidates)} candidate sources.",
                candidates[:5],
            )

            self._set_status(job, "reading")
            self._emit(job, "Reader Agents", "working", "Fetching pages, PDFs, and readable text.")
            documents = await self._read_sources(candidates, request)
            selected = self._select_documents(documents, request.max_sources)
            job.citations = [document.source for document in selected]
            self._emit(
                job,
                "Reader Agents",
                "complete" if selected else "warning",
                f"Read {len(selected)} usable sources from {len(candidates)} candidates.",
                job.citations[:5],
            )

            self._set_status(job, "extracting")
            self._emit(job, "Citation Agent", "working", "Extracting source-grounded evidence cards.")
            evidence, evidence_setup = await self._extract_evidence(selected, request)
            setup_required.extend(evidence_setup)
            self._emit(
                job,
                "Citation Agent",
                "complete" if evidence else "warning",
                f"Built {len(evidence)} evidence cards tied to citations.",
                job.citations,
            )

            self._set_status(job, "verifying")
            self._emit(job, "Final Boss", "working", "Checking citation coverage, source quality, and gaps.")
            critic_notes = self._critic_notes(selected, evidence, setup_required, request)

            self._set_status(job, "writing")
            self._emit(job, "Writer Agent", "working", "Writing the cited Markdown research report.")
            report_markdown, report_setup = await self._write_report(request, selected, evidence, critic_notes)
            setup_required.extend(report_setup)

            confidence = self._confidence(selected, evidence, report_markdown, setup_required, request)
            summary = self._summary_from_report(report_markdown)
            job.summary = summary
            job.detailed_answer = report_markdown
            job.confidence = confidence
            job.critic_notes = self._critic_notes(selected, evidence, setup_required, request)
            job.setup_required = sorted(set(setup_required))

            if save_report:
                response = self.to_research_response(job)
                job.report = self.reports.save_research_report(request, response)

            self._emit(job, "Final Boss", "complete", f"Quality check complete at {round(confidence * 100)}% confidence.")
            self._emit(job, "Writer Agent", "complete", "Final Markdown report ready.")
            self._set_status(job, "complete")
            self._emit(job, "Complete", "complete", "Research workflow complete.", job.citations[:5])
        except Exception as exc:
            job.error = str(exc)
            job.setup_required = sorted(set(setup_required))
            self._set_status(job, "error")
            self._emit(job, "Complete", "error", f"Research workflow failed: {exc}")

    def _set_status(self, job: ResearchJobResponse, status: str) -> None:
        job.status = status  # type: ignore[assignment]
        job.updated_at = datetime.utcnow()

    def _pro_model(self) -> str:
        model_selector = getattr(self.llm, "model_for_profile", None)
        return model_selector("pro") if callable(model_selector) else self.settings.resolved_cerebras_pro_model

    def _emit(self, job: ResearchJobResponse, agent: str, status: str, message: str, sources: list[Source] | None = None) -> None:
        job.events.append(AgentEvent(agent=agent, status=status, message=message, sources=sources or []))  # type: ignore[arg-type]
        job.updated_at = datetime.utcnow()

    def _plan_queries(self, request: ResearchRequest) -> list[str]:
        topic = re.sub(r"\s+", " ", request.topic).strip()
        year = datetime.utcnow().year
        queries = [
            topic,
            f"{topic} latest {year}",
            f"{topic} latest developments {year} report",
            f"{topic} official documentation announcement",
            f"{topic} statistics data trends {year}",
            f"{topic} comparison benchmark analysis",
            f"{topic} limitations risks criticism",
            f"{topic} expert analysis market research",
        ]
        if request.source_mode in {"academic", "mixed"}:
            queries.extend([f"{topic} research paper", f"{topic} literature review"])
        if request.source_policy == "academic_first":
            queries = [f"{topic} research paper", f"{topic} literature review", *queries]
        return self._dedupe_strings(queries)[:10]

    async def _discover_sources(self, request: ResearchRequest, queries: list[str]) -> tuple[list[Source], list[str]]:
        setup_required: list[str] = []
        discovered: list[Source] = []
        max_candidates = request.max_candidates
        per_query = max(3, min(8, max_candidates // max(1, len(queries)) + 1))

        if request.source_mode in {"web", "mixed"}:
            web_results = await asyncio.gather(*(self.search.search_web(query, max_results=per_query) for query in queries))
            for sources, setup in web_results:
                discovered.extend(sources)
                setup_required.extend(setup)

        if request.source_mode in {"academic", "mixed"}:
            academic_count = max(5, min(12, request.max_sources))
            academic_sources, academic_setup = await self.search.search_academic(request.topic, max_results=academic_count)
            discovered.extend(academic_sources)
            setup_required.extend(academic_setup)

        deduped = self._dedupe_sources(discovered)
        filtered = [source for source in deduped if not self._is_rejected_source(source)]
        ranked = sorted(filtered, key=lambda source: self._candidate_score(source, request), reverse=True)
        return ranked[:max_candidates], sorted(set(setup_required))

    async def _read_sources(self, sources: list[Source], request: ResearchRequest) -> list[SourceDocument]:
        semaphore = asyncio.Semaphore(6)

        async def read_one(source: Source) -> SourceDocument | None:
            async with semaphore:
                return await self._fetch_source_document(source, request)

        documents = await asyncio.gather(*(read_one(source) for source in sources))
        return [document for document in documents if document]

    async def _fetch_source_document(self, source: Source, request: ResearchRequest) -> SourceDocument | None:
        url = source.url.strip()
        domain = self._domain_for_url(url)
        if not url.lower().startswith(("http://", "https://")):
            return self._snippet_document(source, domain, "snippet")

        try:
            async with httpx.AsyncClient(timeout=18, follow_redirects=True, headers={"User-Agent": "AstraResearchBot/1.0"}) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                raw = response.content
                published_at = self._published_date(response.text if "html" in content_type else "")
                if "pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf"):
                    text = self._extract_pdf(raw)
                else:
                    text = self._extract_html(response.text)
        except Exception:
            return self._snippet_document(source, domain, "fetch_failed")

        clean_text = self._clean_text(text)
        if len(clean_text) < 600:
            return self._snippet_document(source, domain, "low_text")

        updated = source.model_copy(
            update={
                "snippet": self._snippet(clean_text),
                "title": self._repair_mojibake(source.title),
                "domain": domain,
                "published_at": published_at,
                "fetched_chars": len(clean_text),
                "quality_score": self._quality_score(source, clean_text, "read", request),
                "extraction_status": "read",
            }
        )
        return SourceDocument(source=updated, text=clean_text[:30000])

    def _snippet_document(self, source: Source, domain: str, status: str) -> SourceDocument | None:
        text = self._clean_text(source.snippet)
        if len(text) < 80:
            return None
        updated = source.model_copy(
            update={
                "domain": domain,
                "title": self._repair_mojibake(source.title),
                "snippet": self._snippet(text),
                "fetched_chars": len(text),
                "quality_score": self._quality_score(source, text, status, ResearchRequest(topic="fallback")),
                "extraction_status": status,
            }
        )
        return SourceDocument(source=updated, text=text)

    def _select_documents(self, documents: list[SourceDocument], max_sources: int) -> list[SourceDocument]:
        seen_text: set[str] = set()
        unique: list[SourceDocument] = []
        for document in sorted(documents, key=lambda item: (item.source.quality_score, item.source.fetched_chars), reverse=True):
            if self._is_rejected_source(document.source) or document.source.quality_score < 0.35:
                continue
            fingerprint = self._text_fingerprint(document.text)
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            unique.append(document)
            if len(unique) >= max_sources:
                break
        return unique

    async def _extract_evidence(self, documents: list[SourceDocument], request: ResearchRequest) -> tuple[list[EvidenceCard], list[str]]:
        local = self._local_evidence(documents, request)
        if not documents:
            return [], []

        evidence_input = "\n\n".join(
            f"SOURCE {index}: {document.source.title}\nURL: {document.source.url}\nTEXT:\n{document.text[:1800]}"
            for index, document in enumerate(documents[:10], start=1)
        )
        system_prompt = (
            "You extract source-grounded research evidence. Return JSON only: "
            "[{\"source_index\":1,\"claim\":\"...\",\"support\":\"...\",\"relevance\":0.0}]. "
            "Every item must use only the provided source text."
        )
        user_prompt = f"Topic: {request.topic}\nExtract 8-18 strong evidence cards from these sources:\n\n{evidence_input}"
        raw, setup = await self.llm.complete(system_prompt, user_prompt, model=self._pro_model())
        parsed = self._parse_evidence(raw, len(documents))
        return (parsed or local)[:24], setup

    def _local_evidence(self, documents: list[SourceDocument], request: ResearchRequest) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        terms = set(re.findall(r"[a-z0-9]{4,}", request.topic.lower()))
        for index, document in enumerate(documents, start=1):
            for sentence in self._best_sentences(document.text, terms, limit=2):
                cards.append(EvidenceCard(source_index=index, claim=sentence[:260], support=sentence[:420], relevance=0.62))
        return cards[:24]

    async def _write_report(
        self,
        request: ResearchRequest,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard],
        critic_notes: list[str],
    ) -> tuple[str, list[str]]:
        sources_block = "\n".join(
            f"[{index}] {document.source.title} - {document.source.url} ({document.source.provider}, {document.source.extraction_status})"
            for index, document in enumerate(documents, start=1)
        )
        evidence_block = "\n".join(
            f"- [{card.source_index}] {card.claim} Support: {card.support}"
            for card in evidence
        )
        notes_block = "\n".join(f"- {note}" for note in critic_notes)
        system_prompt = (
            "You are Astra's senior research analyst. Write a rigorous Markdown research report. "
            "Use inline numeric citations like [1]. Do not make unsupported claims. "
            "Do not invent dates, authors, recipients, memo headers, or metadata. "
            "Do not include Date, To, From, or Subject lines. "
            "Use the current generated metadata supplied by the app, not your own date. "
            "Finish every required section before writing References. "
            "Never end a section mid-sentence or with an open bracket. "
            "For laws, bills, regulations, and policies, say 'introduced', 'proposed', 'pending', or 'if enacted' "
            "unless the source explicitly says enacted, effective, or in force. "
            "For ideation topics, avoid generic advice and include concrete tactical plays with who to target, "
            "hidden demand signals, outreach angles, first-week experiments, pricing ideas, and risks. "
            "Include exactly these H2 sections: Executive Summary, Background, Key Findings, Detailed Analysis, "
            "Tactical Playbook, Latest Developments, Limitations and Risks, Conclusion, References."
        )
        user_prompt = (
            f"Topic: {request.topic}\nDepth: {request.depth}\nSource policy: {request.source_policy}\n\n"
            f"Sources:\n{sources_block or 'No readable sources.'}\n\n"
            f"Evidence:\n{evidence_block or 'No source-grounded evidence was extracted.'}\n\n"
            f"Quality notes:\n{notes_block or '- No notes.'}\n\n"
            "Write the final report now. References must map citation numbers to source URLs. "
            "Use cautious wording such as 'the source says' for blog/editorial sources; reserve 'data shows' "
            "for statistical, official, or primary research sources."
        )
        raw, setup = await self.llm.complete(system_prompt, user_prompt, model=self._pro_model())
        if self._report_is_usable(raw, documents):
            repaired = self._post_process_report(raw, request, documents, evidence, critic_notes)
            if not self.validator.validate(repaired, len(documents)):
                return repaired, setup

        fallback = self._post_process_report(self._fallback_report(request, documents, evidence, critic_notes), request, documents, evidence, critic_notes)
        fallback_issues = self.validator.validate(fallback, len(documents))
        if fallback_issues:
            raise ValueError(f"Report quality validation failed after fallback: {', '.join(fallback_issues)}")
        return fallback, setup

    def _fallback_report(
        self,
        request: ResearchRequest,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard],
        critic_notes: list[str],
    ) -> str:
        title = request.topic.strip()
        cited_overview = self._evidence_sentence(evidence, 0)
        findings = "\n".join(f"- {card.claim} [{card.source_index}]" for card in evidence[:8]) or "- No source-grounded findings were available."
        analysis = "\n\n".join(
            f"Source [{index}] contributes: {self._snippet(document.text, 700)} [{index}]"
            for index, document in enumerate(documents[:8], start=1)
        ) or "Astra could not fetch enough readable source text for detailed analysis."
        tactical = self._fallback_tactical_plays(evidence, documents, limit=6) or "- Build a small source-backed experiment first; source coverage was too thin for stronger tactical recommendations."
        limitations = "\n".join(f"- {note}" for note in critic_notes) or "- No major limitations detected."
        references = self._references(documents)
        return (
            f"# {title}\n\n"
            f"## Executive Summary\n{cited_overview or 'Astra prepared a research report, but source coverage was limited.'}\n\n"
            f"## Background\nThis report prioritizes recent web evidence and uses academic sources as supporting context when available.\n\n"
            f"## Key Findings\n{findings}\n\n"
            f"## Detailed Analysis\n{analysis}\n\n"
            f"## Tactical Playbook\n{tactical}\n\n"
            f"## Latest Developments\nRecent-source coverage is reflected through latest-web-first queries and higher ranking for current-year source matches.\n\n"
            f"## Limitations and Risks\n{limitations}\n\n"
            f"## Conclusion\nThe strongest conclusions are the ones directly tied to the cited sources above. Treat uncited areas as gaps for follow-up research.\n\n"
            f"## References\n{references}"
        )

    def _critic_notes(
        self,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard],
        setup_required: list[str],
        request: ResearchRequest,
    ) -> list[str]:
        notes: list[str] = []
        if not documents:
            notes.append("No readable live sources were retrieved; check network access or search provider setup.")
        elif len(documents) < min(8, request.max_sources):
            notes.append(f"Only {len(documents)} readable sources were available, below the requested depth.")
        if len({document.source.domain for document in documents if document.source.domain}) < min(4, len(documents)):
            notes.append("Source diversity is limited; several findings may come from the same domain family.")
        if not evidence:
            notes.append("No source-grounded evidence cards were extracted, so report confidence is limited.")
        if "CEREBRAS_API_KEY" in setup_required:
            notes.append("Cerebras key is missing, so Astra used deterministic fallback report structure.")
        if not notes:
            notes.append("Sources were read, evidence was extracted, and citations were organized for the final report.")
        return notes

    def _confidence(
        self,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard],
        report: str,
        setup_required: list[str],
        request: ResearchRequest,
    ) -> float:
        if not documents:
            return 0.12
        domain_count = len({document.source.domain for document in documents if document.source.domain})
        avg_quality = sum(document.source.quality_score for document in documents) / max(1, len(documents))
        citation_ids = {int(match) for match in re.findall(r"\[(\d{1,2})\]", report)}
        citation_coverage = min(1.0, len(citation_ids) / max(1, min(len(documents), 8)))
        score = 0.15
        score += min(0.25, len(documents) / max(1, request.max_sources) * 0.25)
        score += min(0.15, domain_count / 6 * 0.15)
        score += min(0.25, avg_quality * 0.25)
        score += min(0.12, len(evidence) / 16 * 0.12)
        score += citation_coverage * 0.12
        if setup_required:
            score -= 0.12
        return round(max(0.05, min(0.95, score)), 2)

    def _candidate_score(self, source: Source, request: ResearchRequest) -> float:
        text = f"{source.title} {source.snippet} {source.url}".lower()
        score = 0.3
        if source.provider in {"Tavily", "DuckDuckGo"}:
            score += 0.1
        if source.provider in {"OpenAlex", "Semantic Scholar"}:
            score += 0.08
        if request.source_policy == "latest_web_first":
            year = str(datetime.utcnow().year)
            if "latest" in text or year in text:
                score += 0.15
        domain = self._domain_for_url(source.url)
        if domain.endswith(".gov") or domain.endswith(".edu"):
            score += 0.15
        if any(part in domain for part in ["docs.", "developer.", "research.", "who.int", "worldbank.org"]):
            score += 0.12
        if any(part in domain for part in ["reddit.com", "quora.com", "pinterest.", "facebook.", "instagram."]):
            score -= 0.25
        if self._is_rejected_source(source):
            score -= 0.45
        if re.search(r"\b(top|best|ways|tips|things|ultimate guide)\b", text):
            score -= 0.05
        return score

    def _quality_score(self, source: Source, text: str, status: str, request: ResearchRequest) -> float:
        score = self._candidate_score(source, request)
        if status == "read":
            score += 0.25
        elif status == "snippet":
            score += 0.06
        score += min(0.2, len(text) / 12000)
        return round(max(0.05, min(1.0, score)), 3)

    def _dedupe_sources(self, sources: list[Source]) -> list[Source]:
        seen: set[str] = set()
        unique: list[Source] = []
        for source in sources:
            key = self._source_key(source)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(
                source.model_copy(
                    update={
                        "domain": self._domain_for_url(source.url),
                        "title": self._repair_mojibake(source.title),
                        "snippet": self._repair_mojibake(source.snippet),
                    }
                )
            )
        return unique

    def _source_key(self, source: Source) -> str:
        doi = re.search(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", source.url.lower())
        if doi:
            return doi.group(0)
        parsed = urlparse(source.url.strip())
        if parsed.netloc:
            normalized = parsed._replace(query="", fragment="")
            path = normalized.path.rstrip("/") or "/"
            return urlunparse((normalized.scheme.lower(), normalized.netloc.lower().removeprefix("www."), path, "", "", ""))
        return re.sub(r"\W+", "", source.title.lower())[:120]

    def _dedupe_strings(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            key = value.lower()
            if key not in seen:
                seen.add(key)
                unique.append(value)
        return unique

    def _domain_for_url(self, url: str) -> str:
        return urlparse(url).netloc.lower().removeprefix("www.")

    def _is_rejected_source(self, source: Source) -> bool:
        domain = self._domain_for_url(source.url)
        text = f"{source.title} {source.url}".lower()
        rejected_domains = (
            "linkedin.com",
            "youtube.com",
            "youtu.be",
            "reddit.com",
            "quora.com",
            "facebook.com",
            "instagram.com",
            "tiktok.com",
            "pinterest.",
            "medium.com",
            "merriam-webster.com",
            "dictionary.com",
            "thesaurus.com",
            "vocabulary.com",
        )
        if any(part in domain for part in rejected_domains):
            return True
        rejected_patterns = (
            r"\bposted on the topic\b",
            r"\bcomments?\b.*linkedin",
            r"\byoutube\b",
            r"\bpodcast\b",
            r"\bdictionary\b",
            r"\bdefinition\b",
            r"\bmeaning\b",
        )
        return any(re.search(pattern, text) for pattern in rejected_patterns)

    def _extract_html(self, raw_html: str) -> str:
        parser = _ReadableHtmlParser()
        parser.feed(raw_html)
        return parser.text()

    def _extract_pdf(self, raw: bytes) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages[:12])
        except Exception:
            return ""

    def _published_date(self, raw_html: str) -> str | None:
        patterns = [
            r'"datePublished"\s*:\s*"([^"]+)"',
            r'property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
            r'name=["\']date["\'][^>]+content=["\']([^"\']+)["\']',
            r"\b(20\d{2}-\d{2}-\d{2})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw_html, flags=re.IGNORECASE)
            if match:
                return match.group(1)[:32]
        return None

    def _clean_text(self, text: str) -> str:
        clean = self._repair_mojibake(unescape(text or ""))
        clean = re.sub(r"\r", "\n", clean)
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n\s*\n\s*\n+", "\n\n", clean)
        return clean.strip()

    def _repair_mojibake(self, text: str) -> str:
        if not re.search(r"[\u00c3\u00c2\u00e2]", text):
            return text
        best = text
        best_score = self._mojibake_score(text)
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = text.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            score = self._mojibake_score(candidate)
            if score < best_score:
                best = candidate
                best_score = score
        return best

    def _mojibake_score(self, text: str) -> int:
        markers = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u0080", "\u00e2\u20ac\u2122", "\u00e2\u20ac\u0153")
        return sum(text.count(marker) for marker in markers)

    def _snippet(self, text: str, limit: int = 360) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        return clean[:limit].rstrip()

    def _text_fingerprint(self, text: str) -> str:
        words = re.findall(r"[a-z0-9]{4,}", text.lower())
        return " ".join(words[:80])

    def _best_sentences(self, text: str, terms: set[str], limit: int) -> list[str]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(sentence.strip()) > 60]
        scored: list[tuple[float, str]] = []
        for sentence in sentences[:120]:
            lower = sentence.lower()
            score = sum(1 for term in terms if term in lower)
            if re.search(r"\b20\d{2}\b|%|\b\d+(?:\.\d+)?\b", sentence):
                score += 1.5
            scored.append((score, sentence))
        return [sentence for _, sentence in sorted(scored, reverse=True)[:limit]]

    def _parse_evidence(self, raw: str, source_count: int) -> list[EvidenceCard]:
        payload = self._json_payload(raw)
        if not isinstance(payload, list):
            return []
        cards: list[EvidenceCard] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                source_index = int(item.get("source_index", 0))
                relevance = float(item.get("relevance", 0.5))
            except (TypeError, ValueError):
                continue
            claim = self._repair_mojibake(str(item.get("claim") or "")).strip()
            support = self._repair_mojibake(str(item.get("support") or "")).strip()
            if 1 <= source_index <= source_count and claim and support:
                cards.append(EvidenceCard(source_index=source_index, claim=claim[:360], support=support[:520], relevance=max(0.0, min(1.0, relevance))))
        return cards

    def _json_payload(self, raw: str):
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence.group(1).strip()
        else:
            match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if match:
                text = match.group(0)
        try:
            return json.loads(text)
        except Exception:
            return None

    def _report_is_usable(self, report: str, documents: list[SourceDocument]) -> bool:
        if not report or len(report) < 500:
            return False
        required = ["Executive Summary", "Background", "Key Findings", "Detailed Analysis", "Tactical Playbook", "References"]
        if sum(1 for section in required if section.lower() in report.lower()) < 4:
            return False
        if documents and not re.search(r"\[\d+\]", report):
            return False
        return True

    def _post_process_report(
        self,
        report: str,
        request: ResearchRequest,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard] | None = None,
        critic_notes: list[str] | None = None,
    ) -> str:
        clean = self._repair_mojibake(report).strip()
        clean = self._strip_memo_headers(clean)
        clean = self._normalize_report_headings(clean)
        clean = self._repair_pending_law_claims(clean)
        if not clean.startswith("# "):
            clean = f"# {request.topic.strip()}\n\n{clean}"
        clean = self._canonicalize_report(clean, request, documents, evidence or [], critic_notes or [])
        clean = self._remove_invalid_citations(clean, documents)
        clean = self._replace_references(clean, documents)
        clean = self._repair_mojibake(clean)
        return clean.strip()

    def _strip_memo_headers(self, report: str) -> str:
        lines = report.splitlines()
        cleaned: list[str] = []
        seen_section = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{2,6}\s+", stripped):
                seen_section = True
            if not seen_section and re.match(r"^\*{0,2}(date|to|from|subject)\*{0,2}\s*:", stripped, flags=re.IGNORECASE):
                continue
            if not seen_section and stripped in {"---", "***", "___"}:
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _normalize_report_headings(self, report: str) -> str:
        required = [
            "Executive Summary",
            "Background",
            "Key Findings",
            "Detailed Analysis",
            "Tactical Playbook",
            "Latest Developments",
            "Limitations and Risks",
            "Conclusion",
            "References",
        ]
        clean = report
        for heading in required:
            clean = re.sub(rf"^#{{1,6}}\s*{re.escape(heading)}\s*$", f"## {heading}", clean, flags=re.IGNORECASE | re.MULTILINE)
        return clean

    def _canonicalize_report(
        self,
        report: str,
        request: ResearchRequest,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard],
        critic_notes: list[str],
    ) -> str:
        title_match = re.search(r"^#\s+(.+?)\s*$", report, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else request.topic.strip()
        if title.lower() in {"research report", "detailed report", "final report"}:
            title = request.topic.strip()
        sections = self.validator.sections(report)
        parts = [f"# {title}"]
        for section in self.validator.required_sections:
            if section == "References":
                continue
            body = sections.get(section.lower(), "").strip()
            if not self._section_is_acceptable(section, body):
                body = self._fallback_section(section, request, documents, evidence, critic_notes)
            parts.append(f"## {section}\n{body.strip()}")
        parts.append("## References\n" + self._references(documents[: min(5, len(documents))]))
        return "\n\n".join(parts)

    def _section_is_acceptable(self, section: str, body: str) -> bool:
        clean = body.strip()
        if len(re.sub(r"\s+", " ", clean)) < 45:
            return False
        if self.validator.has_mojibake(clean) or self.validator.has_dangling_text(f"## {section}\n{clean}"):
            return False
        if section == "Tactical Playbook":
            required_terms = ["client", "signal", "outreach", "week", "pricing", "risk"]
            required_labels = ["client type", "hidden demand signal", "outreach angle", "one-week experiment", "pricing idea", "risk"]
            haystack = clean.lower()
            return sum(1 for term in required_terms if term in haystack) >= 4 and all(label in haystack for label in required_labels)
        return True

    def _fallback_section(
        self,
        section: str,
        request: ResearchRequest,
        documents: list[SourceDocument],
        evidence: list[EvidenceCard],
        critic_notes: list[str],
    ) -> str:
        citation = "[1]" if documents else ""
        if section == "Executive Summary":
            overview = self._evidence_sentence(evidence, 0)
            if overview:
                return (
                    f"{overview} This is treated as the report's strongest source-backed starting point, "
                    "with recommendations framed as experiments until further evidence confirms demand."
                )
            return f"Astra prepared a source-grounded report for {request.topic}. The available evidence is limited, so conclusions are framed cautiously {citation}.".strip()
        if section == "Background":
            return f"This report prioritizes readable recent web sources and uses each source only for claims it can support. When sources describe bills, policies, or regulations, the report treats them as pending unless the source explicitly says they are enacted or in effect {citation}.".strip()
        if section == "Key Findings":
            findings = "\n".join(f"- {card.claim} [{card.source_index}]" for card in evidence[:6])
            return findings or "- Source coverage was too thin for strong findings; use the tactical playbook as a starting hypothesis rather than settled evidence."
        if section == "Detailed Analysis":
            analysis = "\n\n".join(
                f"Source [{index}] contributes this usable evidence: {self._snippet(document.text, 520)} [{index}]"
                for index, document in enumerate(documents[:5], start=1)
            )
            return analysis or "Astra could not fetch enough readable source text for detailed analysis, so the report avoids unsupported claims."
        if section == "Tactical Playbook":
            plays = self._fallback_tactical_plays(evidence, documents, limit=4)
            return plays or self._default_tactical_playbook(documents)
        if section == "Latest Developments":
            return f"The newest usable sources are treated as signals for where demand may be moving, not proof of guaranteed opportunity. If a source discusses proposed legislation or regulation, the safe interpretation is pending or introduced until an enacted/effective source confirms otherwise {citation}.".strip()
        if section == "Limitations and Risks":
            notes = "\n".join(f"- {note}" for note in critic_notes)
            return notes or "- Source quality varies across web results, so tactical recommendations should be tested with small experiments before larger commitments.\n- Claims about legal or regulatory requirements need confirmation from primary legal sources before being sold as compliance advice."
        if section == "Conclusion":
            return "The report is demo-ready only when the tactical recommendations remain tied to cited evidence, the limitations are explicit, and no legal or market claim is overstated. Use the playbook as a short experiment map, then update it with real outreach results."
        return ""

    def _default_tactical_playbook(self, documents: list[SourceDocument]) -> str:
        anchor = " [1]" if documents else ""
        return (
            "1. **Source-backed micro-offer**\n"
            f"   - Client type: small teams already showing the source-backed pain signal{anchor}.\n"
            "   - Hidden demand signal: recent hiring, public complaints, launch deadlines, compliance pages, or repeated manual workflows.\n"
            "   - Outreach angle: send one specific observation, a screenshot of the bottleneck, and an offer to run a fixed-scope experiment.\n"
            "   - One-week experiment: deliver an audit, prototype, teardown, or workflow map instead of selling vague hours.\n"
            "   - Pricing idea: quote a small fixed fee for the first experiment and a retainer only after measurable value appears.\n"
            "   - Risk: validate the pain with a short call before building anything."
        )

    def _repair_pending_law_claims(self, report: str) -> str:
        law_context = r"(bill|act|h\.r\.|hr\s*\d+|legislation|regulation|policy)"
        pending_context = r"(introduced|proposed|pending|referred|would|if enacted)"
        if not re.search(law_context, report, flags=re.IGNORECASE) or not re.search(pending_context, report, flags=re.IGNORECASE):
            return report
        replacements = [
            (r"\bnow in effect\b", "introduced or pending"),
            (r"\bis in effect\b", "is introduced or pending"),
            (r"\bhas taken effect\b", "has been introduced or proposed"),
            (r"\bmandates developers\b", "would require developers if enacted"),
            (r"\brequires developers\b", "would require developers if enacted"),
            (r"\bis now law\b", "has been introduced as legislation"),
        ]
        clean = report
        for pattern, replacement in replacements:
            clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
        return clean

    def _replace_references(self, report: str, documents: list[SourceDocument]) -> str:
        without_refs = re.split(r"^## References\s*$", report, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0].rstrip()
        cited = sorted({int(match) for match in re.findall(r"\[(\d{1,2})\]", without_refs)})
        valid_ids = [index for index in cited if 1 <= index <= len(documents)]
        if valid_ids:
            renumber = {old_id: new_id for new_id, old_id in enumerate(valid_ids, start=1)}

            def replace(match: re.Match[str]) -> str:
                old_id = int(match.group(1))
                return f"[{renumber[old_id]}]" if old_id in renumber else ""

            body = re.sub(r"\[(\d{1,2})\]", replace, without_refs)
            reference_documents = [documents[index - 1] for index in valid_ids]
            return body + "\n\n## References\n" + self._references(reference_documents)
        reference_documents = documents[: min(5, len(documents))]
        return without_refs + "\n\n## References\n" + self._references(reference_documents)

    def _remove_invalid_citations(self, report: str, documents: list[SourceDocument]) -> str:
        max_id = len(documents)

        def replace(match: re.Match[str]) -> str:
            citation_id = int(match.group(1))
            return match.group(0) if 1 <= citation_id <= max_id else ""

        return re.sub(r"\[(\d{1,2})\]", replace, report)

    def _references(self, documents: list[SourceDocument], ids: list[int] | None = None) -> str:
        if not documents:
            return "No live references were available."
        numbers = ids if ids else list(range(1, len(documents) + 1))
        return "\n".join(
            f"{number}. [{self._repair_mojibake(document.source.title)}]({document.source.url}) - {document.source.provider}"
            for number, document in zip(numbers, documents)
        )

    def _fallback_tactical_plays(self, evidence: list[EvidenceCard], documents: list[SourceDocument], limit: int) -> str:
        plays: list[str] = []
        seen: set[str] = set()
        for card in evidence:
            document = documents[card.source_index - 1] if 1 <= card.source_index <= len(documents) else None
            profile = self._tactical_profile(card, document)
            if profile["name"] in seen:
                continue
            seen.add(profile["name"])
            plays.append(self._format_tactical_play(profile, card, len(plays) + 1))
            if len(plays) >= limit:
                break
        return "\n\n".join(plays)

    def _fallback_tactical_play(self, card: EvidenceCard, index: int, documents: list[SourceDocument] | None = None) -> str:
        document = None
        if documents and 1 <= card.source_index <= len(documents):
            document = documents[card.source_index - 1]
        profile = self._tactical_profile(card, document)
        return self._format_tactical_play(profile, card, index)

    def _format_tactical_play(self, profile: dict[str, str], card: EvidenceCard, index: int) -> str:
        return (
            f"{index}. **{profile['name']}**: {profile['hook']} [{card.source_index}]\n"
            f"   - Client type: {profile['client_type']}\n"
            f"   - Hidden demand signal: {profile['hidden_signal']}\n"
            f"   - Outreach angle: {profile['outreach']}\n"
            f"   - One-week experiment: {profile['experiment']}\n"
            f"   - Pricing idea: {profile['pricing']}\n"
            f"   - Risk: {profile['risk']}"
        )

    def _tactical_profile(self, card: EvidenceCard, document: SourceDocument | None) -> dict[str, str]:
        claim_text = f"{card.claim} {card.support}".lower()
        source_text = claim_text
        if document:
            source_text += f" {document.source.title} {document.source.snippet} {document.text[:1200]}".lower()
        if any(term in source_text for term in ["agency", "agencies", "digital agency", "client acquisition", "freelance", "service provider"]):
            return {
                "name": "Agency margin-leak audit",
                "hook": "Find agencies selling labor-heavy deliverables and offer one agent that protects their margin on a live client workflow.",
                "client_type": "small digital agencies, productized service shops, or freelancers with repeatable research, reporting, or QA work.",
                "hidden_signal": "their case studies mention manual reporting, campaign setup, lead research, audits, or weekly client updates.",
                "outreach": "send a one-page teardown of one public deliverable and show which steps could be agent-assisted without changing their offer.",
                "experiment": "automate one internal task for a real client account using dummy data first, then compare time saved after one week.",
                "pricing": "$200-$500 for the teardown and prototype; $500-$2,000/month if it becomes part of their delivery workflow.",
                "risk": "agency owners may fear commoditizing their service, so position the agent as margin protection rather than replacement.",
            }
        if any(term in claim_text for term in ["health", "clinic", "patient", "prior authorization", "insurance", "medical"]):
            return {
                "name": "Back-office queue relief",
                "hook": "Turn visible healthcare admin friction into a privacy-bounded workflow audit instead of a generic automation pitch.",
                "client_type": "small clinics, billing agencies, therapy practices, or healthcare administrators with repetitive document queues.",
                "hidden_signal": "job posts, patient reviews, or website copy mention intake delays, insurance paperwork, or prior authorization friction.",
                "outreach": "offer a private workflow map that identifies which forms, calls, and follow-ups can be safely assisted by an agent.",
                "experiment": "shadow one non-sensitive sample workflow and deliver a prototype checklist agent with dummy data only.",
                "pricing": "$250-$750 for the workflow audit; implementation only after privacy boundaries and approval steps are clear.",
                "risk": "health data is sensitive, so avoid touching protected data until the client confirms compliance requirements.",
            }
        if any(term in claim_text for term in ["real estate", "property", "broker", "cre", "deal", "lease"]):
            return {
                "name": "Deal-room compression audit",
                "hook": "Sell faster document triage for operators who already publish or review dense property packets.",
                "client_type": "small commercial brokers, property investors, or local real-estate operators reviewing long PDFs and comps.",
                "hidden_signal": "they publish offering memoranda, market reports, or listings that require manual extraction before decisions.",
                "outreach": "send a sample red-flag table from one public listing and offer to turn their next packet into an investor brief.",
                "experiment": "process one public packet into rent-roll questions, comp gaps, and a diligence checklist within a week.",
                "pricing": "$100-$300 per packet at first, then per-deal or monthly pricing once the format repeats.",
                "risk": "source documents can be incomplete, so position the output as triage rather than investment advice.",
            }
        if any(term in claim_text for term in ["support", "ticket", "resolution", "customer", "help desk", "e-commerce", "ecommerce"]):
            return {
                "name": "Ticket-drain bounty",
                "hook": "Find a repeated public support problem and sell one measured resolution flow, not a chatbot.",
                "client_type": "small SaaS teams, ecommerce shops, or agencies with public help centers and repeated support questions.",
                "hidden_signal": "their help docs, reviews, or changelogs reveal repeated issues that a triage agent could resolve faster.",
                "outreach": "send five support questions copied from public docs and show the answer gaps a lightweight agent could close.",
                "experiment": "build a private triage script for one recurring issue and measure resolution time over a week.",
                "pricing": "$50-$150 per documented resolution flow or a small per-ticket bonus for verified deflection.",
                "risk": "the agent can answer stale information, so require human review for refunds, account changes, and edge cases.",
            }
        if any(term in claim_text for term in ["legal", "law firm", "attorney", "lawyer", "geo", "seo"]):
            return {
                "name": "AI-answer visibility ambush",
                "hook": "Use AI search visibility as a sharp client-acquisition wedge for firms already competing on trust.",
                "client_type": "small law firms, niche consultants, or local experts who depend on being found before a buyer calls.",
                "hidden_signal": "their site has strong service pages, but AI answers or comparison searches do not mention them.",
                "outreach": "send three buyer-style prompts, show which competitors appear, and offer a paid citation-readiness teardown.",
                "experiment": "build a one-page GEO audit with missing FAQs, source gaps, schema fixes, and three answer-ready content blocks.",
                "pricing": "$150-$400 for the audit, then $600-$1,500 to implement the highest-impact pages.",
                "risk": "do not promise rankings or legal outcomes; sell the audit and implementation work, not guaranteed AI citations.",
            }
        if any(term in source_text for term in ["monetiz", "pricing", "usage", "credit", "subscription", "revenue", "per resolution"]):
            return {
                "name": "Outcome-metered agent counter",
                "hook": "Sell a tiny paid meter around one repeated outcome instead of trying to license a whole product.",
                "client_type": "teams with repeated low-ticket actions such as lead enrichment, document summaries, support answers, or QA checks.",
                "hidden_signal": "they already count tickets, leads, documents, calls, or reviews, which gives you a natural unit to price against.",
                "outreach": "offer to run ten public or dummy examples and price only the successful outputs they would actually use.",
                "experiment": "deliver a spreadsheet of before/after outcomes for one workflow and ask which rows would have been worth paying for.",
                "pricing": "$0.50-$5 per accepted output, or a $99-$299 minimum pilot so the test is not unpaid labor.",
                "risk": "per-output pricing can become support-heavy, so define acceptance criteria and revision limits before the pilot starts.",
            }
        if any(term in source_text for term in ["health", "clinic", "patient", "prior authorization", "insurance", "medical"]):
            return {
                "name": "Back-office queue relief",
                "hook": "Turn visible healthcare admin friction into a privacy-bounded workflow audit instead of a generic automation pitch.",
                "client_type": "small clinics, billing agencies, therapy practices, or healthcare administrators with repetitive document queues.",
                "hidden_signal": "job posts, patient reviews, or website copy mention intake delays, insurance paperwork, or prior authorization friction.",
                "outreach": "offer a private workflow map that identifies which forms, calls, and follow-ups can be safely assisted by an agent.",
                "experiment": "shadow one non-sensitive sample workflow and deliver a prototype checklist agent with dummy data only.",
                "pricing": "$250-$750 for the workflow audit; implementation only after privacy boundaries and approval steps are clear.",
                "risk": "health data is sensitive, so avoid touching protected data until the client confirms compliance requirements.",
            }
        if any(term in source_text for term in ["real estate", "property", "broker", "cre", "deal", "lease"]):
            return {
                "name": "Deal-room compression audit",
                "hook": "Sell faster document triage for operators who already publish or review dense property packets.",
                "client_type": "small commercial brokers, property investors, or local real-estate operators reviewing long PDFs and comps.",
                "hidden_signal": "they publish offering memoranda, market reports, or listings that require manual extraction before decisions.",
                "outreach": "send a sample red-flag table from one public listing and offer to turn their next packet into an investor brief.",
                "experiment": "process one public packet into rent-roll questions, comp gaps, and a diligence checklist within a week.",
                "pricing": "$100-$300 per packet at first, then per-deal or monthly pricing once the format repeats.",
                "risk": "source documents can be incomplete, so position the output as triage rather than investment advice.",
            }
        if any(term in source_text for term in ["support", "ticket", "resolution", "customer", "help desk", "e-commerce", "ecommerce"]):
            return {
                "name": "Ticket-drain bounty",
                "hook": "Find a repeated public support problem and sell one measured resolution flow, not a chatbot.",
                "client_type": "small SaaS teams, ecommerce shops, or agencies with public help centers and repeated support questions.",
                "hidden_signal": "their help docs, reviews, or changelogs reveal repeated issues that a triage agent could resolve faster.",
                "outreach": "send five support questions copied from public docs and show the answer gaps a lightweight agent could close.",
                "experiment": "build a private triage script for one recurring issue and measure resolution time over a week.",
                "pricing": "$50-$150 per documented resolution flow or a small per-ticket bonus for verified deflection.",
                "risk": "the agent can answer stale information, so require human review for refunds, account changes, and edge cases.",
            }
        if any(term in source_text for term in ["legal", "law firm", "attorney", "lawyer", "geo", "seo", "marketing"]):
            return {
                "name": "AI-answer visibility ambush",
                "hook": "Use AI search visibility as a sharp client-acquisition wedge for firms already competing on trust.",
                "client_type": "small law firms, niche consultants, or local experts who depend on being found before a buyer calls.",
                "hidden_signal": "their site has strong service pages, but AI answers or comparison searches do not mention them.",
                "outreach": "send three buyer-style prompts, show which competitors appear, and offer a paid citation-readiness teardown.",
                "experiment": "build a one-page GEO audit with missing FAQs, source gaps, schema fixes, and three answer-ready content blocks.",
                "pricing": "$150-$400 for the audit, then $600-$1,500 to implement the highest-impact pages.",
                "risk": "do not promise rankings or legal outcomes; sell the audit and implementation work, not guaranteed AI citations.",
            }
        return {
            "name": "Public-signal micro-bounty",
            "hook": "Convert one public clue into a paid micro-experiment before pitching a larger build.",
            "client_type": "operators, agencies, creators, or local businesses whose public pages show repeated manual work.",
            "hidden_signal": "recent hiring posts, support pages, launch notes, compliance pages, or reviews expose a task they keep repeating.",
            "outreach": "send one specific observation from their public footprint and offer a paid one-week teardown instead of asking for a meeting.",
            "experiment": "deliver a tiny agent prototype, workflow map, or before/after audit using public or dummy data only.",
            "pricing": "$75-$250 for the first experiment; convert to $500-$2,000 only after the client confirms saved time or revenue.",
            "risk": "public signals do not always mean budget, so validate urgency before building anything unpaid.",
        }

    def _evidence_sentence(self, evidence: list[EvidenceCard], index: int) -> str:
        if not evidence:
            return ""
        card = evidence[min(index, len(evidence) - 1)]
        return f"{card.claim} [{card.source_index}]"

    def _summary_from_report(self, report: str) -> str:
        match = re.search(r"## Executive Summary\s*(.*?)(?:\n## |\Z)", report, flags=re.DOTALL | re.IGNORECASE)
        text = match.group(1).strip() if match else report.strip()
        first = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text))[0].strip()
        return first[:300] if first else "Research report completed."
