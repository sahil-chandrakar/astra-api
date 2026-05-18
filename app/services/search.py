from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import Settings
from app.models import Source


class SearchService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def search_web(self, query: str, max_results: int = 5) -> tuple[list[Source], list[str]]:
        if self.settings.has_tavily:
            tavily_results = await self._search_tavily(query, max_results)
            if tavily_results:
                return tavily_results, []

        fallback_results = await self._search_duckduckgo(query, max_results)
        setup = [] if fallback_results else ["TAVILY_API_KEY"]
        return fallback_results, setup

    async def search_academic(self, query: str, max_results: int = 5) -> tuple[list[Source], list[str]]:
        openalex_task = self._search_openalex(query, max_results)
        semantic_task = self._search_semantic_scholar(query, max_results)
        openalex, semantic = await asyncio.gather(openalex_task, semantic_task)
        merged = self._dedupe_sources([*openalex, *semantic])
        return merged[:max_results], []

    async def _search_tavily(self, query: str, max_results: int) -> list[Source]:
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
        }
        headers = {"Authorization": f"Bearer {self.settings.tavily_api_key}"}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        return [
            Source(
                title=item.get("title") or "Untitled web result",
                url=item.get("url") or "",
                snippet=item.get("content") or "",
                provider="Tavily",
            )
            for item in data.get("results", [])
            if item.get("url")
        ]

    async def _search_openalex(self, query: str, max_results: int) -> list[Source]:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(
                    "https://api.openalex.org/works",
                    params={"search": query, "per-page": max_results},
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        sources: list[Source] = []
        for item in data.get("results", []):
            url = item.get("doi") or item.get("id") or ""
            abstract = self._openalex_abstract(item.get("abstract_inverted_index"))
            sources.append(
                Source(
                    title=item.get("title") or "Untitled academic result",
                    url=url,
                    snippet=abstract[:420],
                    provider="OpenAlex",
                )
            )
        return sources

    async def _search_semantic_scholar(self, query: str, max_results: int) -> list[Source]:
        headers = {}
        if self.settings.semantic_scholar_api_key:
            headers["x-api-key"] = self.settings.semantic_scholar_api_key

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": max_results,
                        "fields": "title,abstract,url,year,authors",
                    },
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        return [
            Source(
                title=item.get("title") or "Untitled paper",
                url=item.get("url") or "",
                snippet=item.get("abstract") or "",
                provider="Semantic Scholar",
            )
            for item in data.get("data", [])
            if item.get("url")
        ]

    async def _search_duckduckgo(self, query: str, max_results: int) -> list[Source]:
        try:
            from duckduckgo_search import DDGS

            def run_search() -> list[dict[str, Any]]:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))

            raw_results = await asyncio.to_thread(run_search)
        except Exception:
            return []

        return [
            Source(
                title=item.get("title") or "Untitled result",
                url=item.get("href") or "",
                snippet=item.get("body") or "",
                provider="DuckDuckGo",
            )
            for item in raw_results
            if item.get("href")
        ]

    def _dedupe_sources(self, sources: list[Source]) -> list[Source]:
        seen: set[str] = set()
        unique: list[Source] = []
        for source in sources:
            key = source.url.lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(source)
        return unique

    def _openalex_abstract(self, inverted: dict[str, list[int]] | None) -> str:
        if not inverted:
            return ""
        words: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            words.extend((position, word) for position in positions)
        return " ".join(word for _, word in sorted(words))
