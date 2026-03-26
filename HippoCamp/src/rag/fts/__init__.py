"""FTS (Full-Text Search) module for BM25 lexical retrieval."""

from .sqlite_fts import SqliteFtsStore

__all__ = ["SqliteFtsStore"]
