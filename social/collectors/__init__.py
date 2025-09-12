"""Social media data collectors package."""

from .x_normalize import XAnalyticsNormalizer

# Optional: twscrape-based collector (exported if importable at runtime)
try:  # pragma: no cover - import guard
    from .twscrape_collector import TweetRecord, TWScrapeCollector

    __all__ = ["XAnalyticsNormalizer", "TWScrapeCollector", "TweetRecord"]
except Exception:  # twscrape not installed
    __all__ = ["XAnalyticsNormalizer"]
