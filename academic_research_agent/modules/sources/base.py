"""Base source interface — all data-source clients inherit from this."""

from abc import ABC, abstractmethod


class BaseSource(ABC):
    """Abstract base for a scholarly-paper data source.

    Subclasses must implement ``name`` and ``search``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'arxiv', 'openalex', 'semantic_scholar', etc."""
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 20,
        **kwargs,
    ) -> list[dict]:
        """Search for papers and return a list of unified paper dicts.

        Parameters
        ----------
        query : str
            Search query string.
        max_results : int
            Maximum number of results to return.
        **kwargs
            Source-specific additional parameters.

        Returns
        -------
        list[dict]
            Paper dicts conforming to ``paper_schema``.
        """
        ...

    def health_check(self) -> dict:
        """Return a dict describing the current status of this source.

        Default implementation returns ``{"ok": True}``.  Override if the
        source does real connectivity checks.
        """
        return {"ok": True}
