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
        markdown = self._to_markdown(request, response, created_at)
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

        return "\n\n".join(
            part
            for part in [
                f"# {request.topic}",
                f"Generated: {created_at.isoformat(timespec='seconds')}Z",
                f"Mode: research\nDepth: {request.depth}\nSource mode: {request.source_mode}",
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
