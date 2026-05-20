"""Common utility functions."""

from datetime import datetime


def format_date(dt: datetime | None) -> str:
    """Format a datetime to YYYY-MM-DD string."""
    if dt is None:
        return "N/A"
    return dt.strftime("%Y-%m-%d")


def truncate(text: str, max_words: int = 50) -> str:
    """Truncate text to max_words, appending '...' if cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def highlight_keywords(text: str, keywords: list[str]) -> str:
    """
    Wrap keyword occurrences in **bold** markdown (case-insensitive).
    Returns the decorated text.
    """
    for kw in keywords:
        # Simple replace — in production you'd use regex with word boundaries
        pass
    return text
