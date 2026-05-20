"""Data-source clients for academic paper retrieval."""

from modules.sources.base import BaseSource
from modules.sources.arxiv_client import ArxivSource
from modules.sources.openalex_client import OpenAlexSource
from modules.sources.semantic_scholar_client import SemanticScholarSource

__all__ = [
    "BaseSource",
    "ArxivSource",
    "OpenAlexSource",
    "SemanticScholarSource",
]
