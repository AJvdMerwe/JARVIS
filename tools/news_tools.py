"""
tools/news_tools.py
────────────────────
LangChain tools for the News Agent:
  • RSSFeedTool     – fetch and summarise articles from RSS feeds.
  • HeadlinesTool   – return the latest headlines across all configured feeds.
  • TopicNewsTool   – search DuckDuckGo News for a specific topic.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Type

import feedparser  # type: ignore
import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; VirtualAssistant/1.0; +https://github.com)"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

class HeadlinesInput(BaseModel):
    max_articles: int = Field(
        default=settings.news_max_articles,
        ge=1, le=50,
        description="Maximum number of headlines to return",
    )
    category: Optional[str] = Field(
        None,
        description="Optional topic filter (e.g. 'technology', 'finance')",
    )


class TopicNewsInput(BaseModel):
    topic: str = Field(..., description="Topic or query to search news for")
    max_results: int = Field(default=5, ge=1, le=20)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_rss(url: str, max_items: int = 10) -> list[dict]:
    """Fetch and parse an RSS feed, returning a list of article dicts."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        feed = feedparser.parse(resp.text)
        articles = []
        for entry in feed.entries[:max_items]:
            published = ""
            if hasattr(entry, "published"):
                published = entry.published
            elif hasattr(entry, "updated"):
                published = entry.updated

            articles.append(
                {
                    "title": getattr(entry, "title", "No title"),
                    "summary": getattr(entry, "summary", "")[:400],
                    "link": getattr(entry, "link", ""),
                    "published": published,
                    "source": feed.feed.get("title", url),
                }
            )
        return articles
    except Exception as exc:
        logger.warning("Failed to fetch RSS '%s': %s", url, exc)
        return []


def _format_articles(articles: list[dict]) -> str:
    if not articles:
        return "No articles found."
    lines = []
    for i, art in enumerate(articles, 1):
        lines.append(
            f"{i}. **{art['title']}**\n"
            f"   Source: {art['source']} | {art['published']}\n"
            f"   {art['summary']}\n"
            f"   URL: {art['link']}\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Tools
# ─────────────────────────────────────────────────────────────────────────────

class HeadlinesTool(BaseTool):
    """Fetch top headlines from all configured RSS feeds."""

    name: str = "get_headlines"
    description: str = (
        "Get the latest news headlines from multiple sources. "
        "Optionally filter by category (e.g. 'technology', 'finance', 'sports')."
    )
    args_schema: Type[BaseModel] = HeadlinesInput

    def _run(
        self,
        max_articles: int = settings.news_max_articles,
        category: Optional[str] = None,
    ) -> str:
        feeds = settings.rss_feed_list
        all_articles: list[dict] = []

        per_feed = max(1, max_articles // len(feeds))
        for feed_url in feeds:
            all_articles.extend(_fetch_rss(feed_url, max_items=per_feed + 2))

        # Simple category filter on title/summary text
        if category:
            kw = category.lower()
            all_articles = [
                a for a in all_articles
                if kw in a["title"].lower() or kw in a["summary"].lower()
            ]

        # Sort by most recent and truncate
        all_articles = all_articles[:max_articles]

        header = f"## Latest Headlines ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\n\n"
        return header + _format_articles(all_articles)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class TopicNewsTool(BaseTool):
    """Search for news articles on a specific topic."""

    name: str = "search_topic_news"
    description: str = (
        "Search for recent news articles about a specific topic or query. "
        "Returns titles, summaries, and URLs."
    )
    args_schema: Type[BaseModel] = TopicNewsInput

    def _run(self, topic: str, max_results: int = 5) -> str:
        try:
            from duckduckgo_search import DDGS  # type: ignore
            # from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.news(topic, max_results=max_results))

            if not results:
                return f"No news found for topic: '{topic}'"

            articles = [
                {
                    "title": r.get("title", ""),
                    "summary": r.get("body", "")[:400],
                    "link": r.get("url", ""),
                    "published": r.get("date", ""),
                    "source": r.get("source", ""),
                }
                for r in results
            ]
            return f"## News: {topic}\n\n" + _format_articles(articles)

        except Exception as exc:
            logger.error("Topic news search failed: %s", exc)
            return f"Could not retrieve news for '{topic}': {exc}"

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


class RSSFeedTool(BaseTool):
    """Fetch articles from a specific RSS feed URL."""

    name: str = "fetch_rss_feed"
    description: str = (
        "Fetch and summarise articles from a specific RSS feed URL. "
        "Provide a valid RSS/Atom feed URL."
    )

    class _Input(BaseModel):
        url: str = Field(..., description="The RSS/Atom feed URL to fetch")
        max_items: int = Field(default=5, ge=1, le=20)

    args_schema: Type[BaseModel] = _Input

    def _run(self, url: str, max_items: int = 5) -> str:
        articles = _fetch_rss(url, max_items)
        return f"## Feed: {url}\n\n" + _format_articles(articles)

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


def get_news_tools() -> list[BaseTool]:
    """Return all news-related tools."""
    return [
        HeadlinesTool(),
        TopicNewsTool(),
        RSSFeedTool(),
    ]
