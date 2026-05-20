import asyncio

import pytest

from app.config import Settings
from app.models import ResearchRequest, ResearchResponse, Source
from app.services.reports import ReportService
from app.services.research import EvidenceCard, ReportQualityValidator, ResearchService, SourceDocument


class FakeSearch:
    async def search_web(self, query: str, max_results: int = 5):
        return [
            Source(title=f"Latest result for {query}", url="https://example.com/research", snippet="latest evidence and 2026 data", provider="DuckDuckGo"),
            Source(title="Duplicate latest result", url="https://example.com/research?utm=1", snippet="duplicate", provider="DuckDuckGo"),
            Source(title="Official update", url="https://docs.example.org/update", snippet="official update details", provider="Tavily"),
        ][:max_results], []

    async def search_academic(self, query: str, max_results: int = 5):
        return [
            Source(title="Academic support paper", url="https://doi.org/10.1234/astra-study", snippet="peer reviewed context", provider="OpenAlex"),
        ], []


class EmptySearch:
    async def search_web(self, query: str, max_results: int = 5):
        return [], []

    async def search_academic(self, query: str, max_results: int = 5):
        return [], []


class FakeLlm:
    def __init__(self):
        self.calls: list[str] = []

    async def complete(self, system_prompt: str, user_prompt: str, model: str | None = None):
        self.calls.append(system_prompt)
        if "Return JSON only" in system_prompt:
            return (
                '[{"source_index":1,"claim":"The topic has fresh 2026 evidence.","support":"Fresh 2026 evidence is present in the source.","relevance":0.9},'
                '{"source_index":2,"claim":"Official documentation provides implementation context.","support":"The official source describes implementation details.","relevance":0.8}]',
                [],
            )
        return (
            "# Agentic Research Mode\n\n"
            "## Executive Summary\nFresh evidence supports the report [1].\n\n"
            "## Background\nThe background is source-grounded [1].\n\n"
            "## Key Findings\n- Official sources add implementation context [2].\n\n"
            "## Detailed Analysis\nThe analysis cites the latest source and official source [1][2].\n\n"
            "## Latest Developments\nThe latest development is covered by recent web evidence [1].\n\n"
            "## Limitations and Risks\nCoverage remains dependent on readable source access [2].\n\n"
            "## Conclusion\nThe report is citation-backed [1].\n\n"
            "## References\n1. https://example.com/research\n2. https://docs.example.org/update",
            [],
        )


class BrokenReportLlm(FakeLlm):
    async def complete(self, system_prompt: str, user_prompt: str, model: str | None = None):
        self.calls.append(system_prompt)
        if "Return JSON only" in system_prompt:
            return (
                '[{"source_index":1,"claim":"Student developers can sell AI-agent workflow audits tied to public business signals.","support":"The source discusses practical AI-agent workflow demand signals.","relevance":0.92},'
                '{"source_index":2,"claim":"Proposed AI legislation should be treated as pending unless enacted.","support":"The source says the bill was introduced and referred.","relevance":0.84}]',
                [],
            )
        return (
            "# Research Report\n\n"
            "**Date:** October 26, 2023\n"
            "**To:** Astra Executive Team\n"
            "**From:** Senior Research Analyst\n"
            "**Subject:** Made-up memo header\n\n"
            "---\n\n"
            "### Executive Summary\n"
            "AI-agent service demand creates odd micro-offer opportunities for student developers, but this draft has mojibake â€” and an invalid citation [1][99]. "
            "The AI Foundation Model Transparency Act is now in effect and requires developers, even though the source context says the bill was introduced and pending [2].\n\n"
            "### Background\n"
            "The topic is best researched through recent web sources, public demand signals, and cautious legal framing when bills are merely proposed [1].\n\n"
            "### Key Findings\n"
            "- Public workflow friction can become a narrow paid diagnostic rather than generic freelance pitching [1].\n"
            "- Pending bills should be described as introduced, proposed, or if enacted rather than live law [2].\n\n"
            "### Detailed Analysis\n"
            "Student developers should look for client pain that is already visible in hiring posts, support docs, launch notes, and manual operations pages [1]. "
            "The analysis should stay source-grounded and avoid treating a proposed act as an active compliance mandate [2].\n\n"
            "### Latest Developments\n"
            "Recent source coverage mentions proposed legislation, but this broken report trails off with risks [\n\n"
            "### References\n"
            "1. stale reference\n"
            "99. invalid reference\n\n"
            "### References\n"
            "Duplicate reference block",
            [],
        )


class FakeFetchResearchService(ResearchService):
    async def _fetch_source_document(self, source: Source, request: ResearchRequest):
        text = (
            f"{source.title} gives detailed evidence about {request.topic}. "
            "It includes 2026 details, implementation context, statistics, limitations, and practical findings. "
            * 20
        )
        updated = source.model_copy(
            update={
                "domain": self._domain_for_url(source.url),
                "fetched_chars": len(text),
                "quality_score": 0.82,
                "extraction_status": "read",
                "snippet": text[:320],
            }
        )
        return SourceDocument(source=updated, text=text)


def build_service(tmp_path, search=None, llm=None):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
        cerebras_pro_model="pro-model",
    )
    return FakeFetchResearchService(settings, llm or FakeLlm(), search or FakeSearch(), ReportService(settings))


@pytest.mark.asyncio
async def test_background_research_job_completes_and_saves_report(tmp_path):
    service = build_service(tmp_path)

    job = service.start_job(ResearchRequest(topic="agentic research mode", max_candidates=12, max_sources=4))

    assert job.status == "queued"
    for _ in range(50):
        if job.status in {"complete", "error"}:
            break
        await asyncio.sleep(0.02)

    assert job.status == "complete"
    assert job.report is not None
    assert "## Executive Summary" in job.report.markdown
    assert "## References" in job.report.markdown
    assert len(job.citations) >= 2
    assert job.confidence > 0.5
    assert [event.agent for event in job.events if event.agent in {"Search Agents", "Reader Agents", "Writer Agent", "Complete"}]


def test_latest_web_first_query_planner_creates_multiple_search_angles(tmp_path):
    service = build_service(tmp_path)

    queries = service._plan_queries(ResearchRequest(topic="multi agent LLM systems"))

    assert len(queries) >= 8
    assert any("latest" in query.lower() for query in queries)
    assert any("official" in query.lower() for query in queries)
    assert any("limitations" in query.lower() for query in queries)
    assert any("research paper" in query.lower() for query in queries)


def test_source_dedupe_uses_normalized_urls_and_doi(tmp_path):
    service = build_service(tmp_path)
    sources = [
        Source(title="A", url="https://example.com/path?utm=1", snippet="", provider="x"),
        Source(title="A copy", url="https://example.com/path", snippet="", provider="x"),
        Source(title="Paper", url="https://doi.org/10.1234/example", snippet="", provider="x"),
        Source(title="Paper copy", url="https://dx.doi.org/10.1234/example", snippet="", provider="x"),
    ]

    deduped = service._dedupe_sources(sources)

    assert len(deduped) == 2


def test_report_post_processing_removes_fake_metadata_mojibake_and_extra_refs(tmp_path):
    service = build_service(tmp_path)
    documents = [
        SourceDocument(Source(title="Primary source", url="https://example.com/one", provider="Tavily", domain="example.com"), "source text one"),
        SourceDocument(Source(title="Second source", url="https://example.org/two", provider="Tavily", domain="example.org"), "source text two"),
    ]
    bad_report = (
        "# Research Report\n\n"
        "**Date:** October 26, 2023\n"
        "**To:** Astra Executive Team\n"
        "**From:** Senior Research Analyst\n"
        "**Subject:** Made-up memo header\n\n"
        "---\n\n"
        "### Executive Summary\nA tactical idea \u00e2\u20ac\u201d with evidence [1][3].\n\n"
        "### Background\nSource-backed background [1].\n\n"
        "### Key Findings\n- Finding [2].\n\n"
        "### Detailed Analysis\nAnalysis [1].\n\n"
        "### Latest Developments\nRecent context [2].\n\n"
        "### Limitations and Risks\nRisk [2].\n\n"
        "### Conclusion\nConclusion [1].\n\n"
        "### References\n1. stale\n2. stale\n3. invalid"
    )

    processed = service._post_process_report(bad_report, ResearchRequest(topic="client acquisition"), documents)

    assert "October 26, 2023" not in processed
    assert "**To:**" not in processed
    assert "\u00e2" not in processed
    assert "[3]" not in processed
    assert "## Executive Summary" in processed
    assert "## Tactical Playbook" in processed
    assert "3. " not in processed.split("## References", 1)[1]


def test_rejects_social_video_and_mediocre_sources(tmp_path):
    service = build_service(tmp_path)

    assert service._is_rejected_source(Source(title="LinkedIn post", url="https://www.linkedin.com/posts/example", provider="Tavily"))
    assert service._is_rejected_source(Source(title="Client Acquisition Video", url="https://youtube.com/watch?v=abc", provider="Tavily"))
    assert service._is_rejected_source(Source(title="Unconventional Definition & Meaning", url="https://www.merriam-webster.com/dictionary/unconventional", provider="Tavily"))
    assert not service._is_rejected_source(Source(title="Official guide", url="https://www.uschamber.com/co/start/business-ideas/top-trending-business-ideas", provider="Tavily"))


@pytest.mark.asyncio
async def test_no_sources_produces_low_confidence_and_setup_notes(tmp_path):
    service = build_service(tmp_path, search=EmptySearch())

    job = await service.run_to_completion(ResearchRequest(topic="unavailable topic", max_candidates=10, max_sources=3), save_report=False)

    assert job.status == "complete"
    assert job.confidence < 0.2
    assert not job.citations
    assert any("No readable live sources" in note for note in job.critic_notes)
    assert "## Limitations and Risks" in job.detailed_answer


def test_quality_validator_rejects_mojibake_and_truncated_sections():
    validator = ReportQualityValidator()
    report = (
        "# Test Report\n\n"
        "## Executive Summary\nThis section contains enough source-grounded detail but also mojibake â€” [1].\n\n"
        "## Background\nThis background has enough detail to be considered present and usable for validation [1].\n\n"
        "## Key Findings\n- A concrete source-backed finding is included for validation [1].\n\n"
        "## Detailed Analysis\nThe analysis is long enough and includes a supported citation from the first source [1].\n\n"
        "## Tactical Playbook\nClient type: operators. Hidden demand signal: manual work. Outreach angle: audit. One-week experiment: teardown. Pricing idea: fixed fee. Risk: weak urgency [1].\n\n"
        "## Latest Developments\nThis section is deliberately broken and ends with risks [\n\n"
        "## Limitations and Risks\nThe report notes limits and avoids overclaiming beyond source coverage [1].\n\n"
        "## Conclusion\nThe conclusion is present and ties recommendations back to cited evidence [1].\n\n"
        "## References\n1. [Source](https://example.com/source) - Tavily"
    )

    issues = validator.validate(report, source_count=1)

    assert "report contains broken text encoding" in issues
    assert "report appears truncated or has dangling punctuation" in issues


def test_fallback_report_passes_quality_validator(tmp_path):
    service = build_service(tmp_path)
    documents = [
        SourceDocument(Source(title="Demand source", url="https://example.com/one", provider="Tavily", domain="example.com"), "source text " * 120),
        SourceDocument(Source(title="Legal source", url="https://example.org/two", provider="Tavily", domain="example.org"), "introduced bill context " * 120),
    ]
    evidence = [
        EvidenceCard(1, "Visible workflow friction can support a narrow AI-agent audit offer.", "The source describes visible workflow friction.", 0.9),
        EvidenceCard(2, "Introduced legislation should be described cautiously until enacted.", "The source says introduced and referred.", 0.8),
    ]

    report = service._post_process_report(
        service._fallback_report(ResearchRequest(topic="student AI agent income"), documents, evidence, []),
        ResearchRequest(topic="student AI agent income"),
        documents,
        evidence,
        [],
    )

    assert service.validator.validate(report, len(documents)) == []
    assert "## Tactical Playbook" in report


@pytest.mark.asyncio
async def test_broken_llm_report_is_repaired_before_saving(tmp_path):
    service = build_service(tmp_path, llm=BrokenReportLlm())

    job = await service.run_to_completion(
        ResearchRequest(
            topic="unconventional ways for student developers to make money with AI agents",
            max_candidates=12,
            max_sources=4,
        ),
        save_report=True,
    )

    assert job.status == "complete"
    assert job.report is not None
    markdown = job.report.markdown
    assert "â" not in markdown
    assert "October 26, 2023" not in markdown
    assert "**To:**" not in markdown
    assert "## Detailed Report\n#" not in markdown
    assert markdown.count("## References") == 1
    for section in ReportQualityValidator.required_sections:
        assert f"## {section}" in markdown
    assert "[99]" not in markdown
    assert "now in effect" not in markdown.lower()
    assert "is now law" not in markdown.lower()
    assert "would require developers if enacted" in markdown.lower() or "introduced or pending" in markdown.lower()


def test_report_service_refuses_invalid_research_markdown(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path / "data"),
        reports_dir=str(tmp_path / "reports"),
        piper_cache_dir=str(tmp_path / "piper"),
    )
    reports = ReportService(settings)
    request = ResearchRequest(topic="bad report")
    bad_markdown = "# Bad Report\n\n## Detailed Report\n# Nested legacy report\n\n## References\n1. [Bad](https://example.com) - Tavily"
    response = ResearchResponse(
        summary="bad",
        detailed_answer=bad_markdown,
        citations=[Source(title="Bad", url="https://example.com", provider="Tavily")],
        confidence=0.1,
    )

    with pytest.raises(ValueError, match="Refusing to save invalid research report"):
        reports.save_research_report(request, response)
