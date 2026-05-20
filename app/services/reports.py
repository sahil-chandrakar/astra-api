import re
from datetime import datetime
from pathlib import Path

from app.config import Settings
from app.models import ResearchReport, ResearchRequest, ResearchResponse


class ReportService:
    def __init__(self, settings: Settings):
        base = Path(settings.reports_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        self.reports_dir = base
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_research_report(self, request: ResearchRequest, response: ResearchResponse) -> ResearchReport:
        created_at = datetime.utcnow()
        slug = self._slugify(request.topic)
        report_id = f"{created_at.strftime('%Y%m%d-%H%M%S')}-{slug}"
        markdown = self._repair_mojibake(self._to_markdown(request, response, created_at))
        self._assert_basic_report_quality(markdown, len(response.citations))
        path = self._path_for(report_id)
        path.write_text(markdown, encoding="utf-8")

        return ResearchReport(
            id=report_id,
            title=request.topic,
            created_at=created_at,
            markdown=markdown,
            download_url=f"/api/reports/{report_id}/download",
        )

    def list_reports(self) -> list[ResearchReport]:
        reports: list[ResearchReport] = []
        for path in sorted(self.reports_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True):
            markdown = path.read_text(encoding="utf-8")
            reports.append(
                ResearchReport(
                    id=path.stem,
                    title=self._title_from_markdown(markdown) or path.stem,
                    created_at=datetime.utcfromtimestamp(path.stat().st_mtime),
                    markdown=markdown,
                    download_url=f"/api/reports/{path.stem}/download",
                )
            )
        return reports

    def get_report_path(self, report_id: str) -> Path | None:
        safe_id = self._safe_report_id(report_id)
        if not safe_id:
            return None
        path = self._path_for(safe_id)
        if not path.exists() or path.parent.resolve() != self.reports_dir.resolve():
            return None
        return path

    def _path_for(self, report_id: str) -> Path:
        return self.reports_dir / f"{report_id}.md"

    def _to_markdown(self, request: ResearchRequest, response: ResearchResponse, created_at: datetime) -> str:
        citations = "\n".join(
            f"{index}. [{source.title}]({source.url}) - {source.provider}\n   {source.snippet}".strip()
            for index, source in enumerate(response.citations, start=1)
        )
        critic_notes = "\n".join(f"- {note}" for note in response.critic_notes)
        setup = "\n".join(f"- {item}" for item in response.setup_required)
        metadata = f"Generated: {created_at.isoformat(timespec='seconds')}Z\nMode: research\nDepth: {request.depth}\nSource mode: {request.source_mode}"

        if "## Executive Summary" in response.detailed_answer and "## References" in response.detailed_answer:
            return response.detailed_answer.strip()

        return "\n\n".join(
            part
            for part in [
                f"# {request.topic}",
                metadata,
                "## Summary\n" + response.summary,
                "## Detailed Report\n" + response.detailed_answer,
                "## Citations\n" + (citations or "No live citations were retrieved."),
                "## Final Boss Notes\n" + (critic_notes or "- No critic notes."),
                "## Setup Notes\n" + (setup or "- All configured providers were available."),
            ]
            if part
        )

    def _title_from_markdown(self, markdown: str) -> str:
        first_line = markdown.splitlines()[0] if markdown.splitlines() else ""
        return first_line.removeprefix("# ").strip()

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return (slug[:64] or "research-report").strip("-")

    def _safe_report_id(self, report_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]", "", report_id).strip(".")

    def _assert_basic_report_quality(self, markdown: str, source_count: int) -> None:
        report = markdown.strip()
        issues: list[str] = []
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
        if len(report) < 500:
            issues.append("report is too short")
        sections = self._sections(report)
        for section in required:
            body = sections.get(section.lower(), "").strip()
            if not body:
                issues.append(f"missing section: {section}")
                continue
            minimum = 20 if section == "References" else 45
            if len(re.sub(r"\s+", " ", body)) < minimum:
                issues.append(f"thin section: {section}")
        if len(re.findall(r"^## References\s*$", report, flags=re.IGNORECASE | re.MULTILINE)) != 1:
            issues.append("References section must appear exactly once")
        if self._has_mojibake(report):
            issues.append("broken text encoding remains")
        if re.search(r"^\s*\*{0,2}(date|to|from|subject)\*{0,2}\s*:", report, flags=re.IGNORECASE | re.MULTILINE):
            issues.append("fake memo metadata remains")
        if re.search(r"^## Detailed Report\s*\n\s*#", report, flags=re.IGNORECASE | re.MULTILINE):
            issues.append("legacy wrapper remains")
        if self._has_dangling_text(report):
            issues.append("report appears truncated")

        body_without_refs = re.split(r"^## References\s*$", report, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[0]
        citation_ids = [int(match) for match in re.findall(r"\[(\d{1,2})\]", body_without_refs)]
        invalid_ids = [citation_id for citation_id in citation_ids if citation_id < 1 or citation_id > source_count]
        if invalid_ids:
            issues.append("invalid citation ids remain")
        references = sections.get("references", "")
        if source_count:
            if not citation_ids:
                issues.append("report has no inline citations")
            if "http://" not in references and "https://" not in references:
                issues.append("references do not contain URLs")
            ref_numbers = {int(match) for match in re.findall(r"^\s*(\d+)\.", references, flags=re.MULTILINE)}
            missing_refs = sorted(set(citation_ids) - ref_numbers)
            if missing_refs:
                issues.append("references do not cover cited sources")
        if self._has_overstated_pending_law(report):
            issues.append("pending law or bill status is overstated")

        if issues:
            raise ValueError("Refusing to save invalid research report: " + "; ".join(issues))

    def _sections(self, markdown: str) -> dict[str, str]:
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
        sections: dict[str, str] = {}
        for index, match in enumerate(matches):
            name = match.group(1).strip().lower()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
            sections[name] = markdown[start:end].strip()
        return sections

    def _has_dangling_text(self, text: str) -> bool:
        stripped = text.strip()
        if stripped.endswith(("[", "(", "{", ":", "-", "##")):
            return True
        if re.search(r"(\[[0-9,\s]*|\([^)]*)$", stripped):
            return True
        for section, body in self._sections(stripped).items():
            if section == "references":
                continue
            if body.strip().endswith(("[", "(", "{", ":", "-")):
                return True
        return False

    def _has_overstated_pending_law(self, text: str) -> bool:
        law_context = r"(bill|act|h\.r\.|hr\s*\d+|legislation|regulation|law)"
        pending_context = r"(introduced|proposed|pending|referred|would|would require|if enacted)"
        enacted_claim = r"(now in effect|is in effect|has taken effect|requires developers|mandates developers|is now law)"
        return bool(
            re.search(law_context, text, flags=re.IGNORECASE)
            and re.search(enacted_claim, text, flags=re.IGNORECASE)
            and re.search(pending_context, text, flags=re.IGNORECASE)
        )

    def _has_mojibake(self, text: str) -> bool:
        markers = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u0080", "\u00e2\u20ac\u2122", "\u00e2\u20ac\u0153")
        return any(marker in text for marker in markers)

    def _repair_mojibake(self, text: str) -> str:
        if not self._has_mojibake(text):
            return text
        best = text
        best_score = sum(text.count(marker) for marker in ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u0080"))
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = text.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            score = sum(candidate.count(marker) for marker in ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00e2\u0080"))
            if score < best_score:
                best = candidate
                best_score = score
        return best
