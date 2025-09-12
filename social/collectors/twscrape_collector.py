"""
TWScrape-based collector for X (Twitter) public data.

Notes
- This module depends on `twscrape` (install via `pip install twscrape`).
- Uses X GraphQL endpoints via twscrape sessions (cookie-auth recommended).
- Designed for prototyping and research; review ToS for production use.

Typing targets Python 3.11 (project default).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TweetRecord:
    """Normalized tweet record shape for downstream processing."""

    id: int
    url: str
    text: str
    created_at: datetime
    username: str
    user_id: int | None
    like_count: int
    retweet_count: int
    reply_count: int
    quote_count: int
    lang: str | None
    hashtags: list[str]
    urls: list[str]


class TWScrapeNotInstalledError(RuntimeError):
    pass


class TWScrapeCollector:
    """Thin wrapper around twscrape for keyword/hashtag and user timeline collection.

    Basic usage:
        collector = TWScrapeCollector()
        async for rec in collector.search("ai productivity", limit=200, min_likes=200):
            ...

    For higher throughput, supply a session database path and login sessions in advance
    using twscrape's own utilities (accounts/cookies). See README for details.
    """

    def __init__(
        self,
        session_db: Path | None = None,
        rate_limit_seconds: float = 1.0,
    ) -> None:
        self._rate_limit_seconds = rate_limit_seconds
        try:
            # Lazy import to avoid hard dependency if unused.
            from twscrape import API  # type: ignore
        except Exception as e:  # pragma: no cover - import guard
            raise TWScrapeNotInstalledError(
                "twscrape is required. Install with `pip install twscrape`."
            ) from e

        # Initialize API (session pool stored in SQLite). If session_db is None,
        # twscrape uses a default in the CWD; we allow passing a custom path.
        if session_db is not None:
            self._api = API(str(session_db))  # type: ignore[name-defined]
        else:
            self._api = API()  # type: ignore[name-defined]

    async def ensure_logged_in(self) -> None:
        """Ensure sessions are logged in (if any are configured).

        If no accounts are configured, twscrape may fall back to guest. For larger
        volumes, pre-load accounts and call `login_all()` once.
        """
        try:
            await self._api.login_all()  # type: ignore[attr-defined]
        except Exception:
            # If there are no accounts configured, login_all may fail; proceed as guest.
            return

    async def search(
        self,
        query: str,
        limit: int = 200,
        days_back: int = 7,
        lang: str | None = "en",
        min_likes: int | None = None,
        include_retweets: bool = False,
    ) -> AsyncIterator[TweetRecord]:
        """Search recent tweets by query with optional filters.

        Query syntax follows X advanced search. We add some common filters:
        - `lang` (defaults to "en")
        - `min_faves` threshold if provided
        - `-is:retweet` unless include_retweets is True
        - date window: since:YYYY-MM-DD
        """

        _query_parts: list[str] = [query]
        if lang:
            _query_parts.append(f"lang:{lang}")
        if min_likes is not None:
            _query_parts.append(f"min_faves:{min_likes}")
        if not include_retweets:
            _query_parts.append("-is:retweet")

        since_date = (datetime.now(UTC) - timedelta(days=days_back)).date()
        _query_parts.append(f"since:{since_date.isoformat()}")

        final_query = " ".join(_query_parts)

        # Use twscrape's search endpoint. We stream and yield normalized records.
        async for tw in self._api.search(final_query, limit=limit):  # type: ignore[attr-defined]
            rec = _to_record(tw)
            yield rec

    async def user_tweets(
        self,
        username: str,
        limit: int = 200,
        days_back: int | None = None,
        include_replies: bool = False,
        include_retweets: bool = False,
    ) -> AsyncIterator[TweetRecord]:
        """Fetch tweets for a user timeline with basic filters."""

        async for tw in self._api.user_tweets(  # type: ignore[attr-defined]
            username,
            limit=limit,
            include_replies=include_replies,
            include_retweets=include_retweets,
        ):
            rec = _to_record(tw)
            if days_back is not None:
                cutoff = datetime.now(UTC) - timedelta(days=days_back)
                if rec.created_at < cutoff:
                    # Stop early if tweets are older than window
                    break
            yield rec

    @staticmethod
    def to_dict(rec: TweetRecord) -> dict[str, Any]:
        return asdict(rec)


def _to_record(tw: Any) -> TweetRecord:
    """Map a twscrape Tweet object to TweetRecord.

    We defensively access attributes to avoid hard dependency on exact schema.
    """

    # Defensive extracts
    tid = getattr(tw, "id", None)
    text = getattr(tw, "rawContent", getattr(tw, "text", ""))
    dt = getattr(tw, "date", None)
    user = getattr(tw, "user", None)
    username = getattr(user, "username", None) or getattr(user, "screen_name", "")
    user_id = getattr(user, "id", None)
    lang = getattr(tw, "lang", None)
    url = getattr(tw, "url", f"https://x.com/{username}/status/{tid}")

    like_count = int(getattr(tw, "likeCount", getattr(tw, "likes", 0)) or 0)
    retweet_count = int(getattr(tw, "retweetCount", getattr(tw, "retweets", 0)) or 0)
    reply_count = int(getattr(tw, "replyCount", getattr(tw, "replies", 0)) or 0)
    quote_count = int(getattr(tw, "quoteCount", getattr(tw, "quotes", 0)) or 0)

    # Entities
    hashtags: list[str] = []
    urls: list[str] = []
    try:
        ents = getattr(tw, "entities", None) or {}
        tags = ents.get("hashtags") or []
        for t in tags:
            tag = t.get("text") if isinstance(t, dict) else getattr(t, "text", None)
            if tag:
                hashtags.append(str(tag))
        url_ents = ents.get("urls") or []
        for u in url_ents:
            expanded = (
                u.get("expanded_url")
                if isinstance(u, dict)
                else getattr(u, "expanded_url", None)
            )
            if expanded:
                urls.append(str(expanded))
    except Exception:
        pass

    created_at = dt if isinstance(dt, datetime) else datetime.now(UTC)

    return TweetRecord(
        id=int(tid),
        url=str(url),
        text=str(text or ""),
        created_at=created_at,
        username=str(username or ""),
        user_id=int(user_id) if user_id is not None else None,
        like_count=like_count,
        retweet_count=retweet_count,
        reply_count=reply_count,
        quote_count=quote_count,
        lang=str(lang) if lang is not None else None,
        hashtags=hashtags,
        urls=urls,
    )
