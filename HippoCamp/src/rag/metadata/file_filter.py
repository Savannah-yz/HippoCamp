"""
File-level filtering utilities for metadata-based retrieval.

Maps extracted metadata (company code, year) to file_ids in the SQLite database.
"""

import logging
import sqlite3
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_file_ids_by_metadata(
    db_path: str,
    company_code: Optional[str],
    years: Optional[List[str]] = None,
    quarter: Optional[str] = None,
) -> List[str]:
    """Find file_ids matching the given company code and/or years.

    Uses LIKE matching on files.name. If both company and years are provided,
    both must match. Multiple years are OR'd (matching any of the years).

    Args:
        db_path: Path to SQLite database.
        company_code: Uppercase company code (e.g., "PEPSICO").
        years: List of 4-digit year strings (e.g., ["2018", "2019", "2020"]).
        quarter: Optional quarter (e.g., "Q2").

    Returns:
        List of matching file_ids. Empty list if no match or both params are None.
    """
    if not company_code and not years:
        return []

    conditions = []
    params: list = []

    if company_code:
        # Match company code at start of filename
        conditions.append("UPPER(name) LIKE ?")
        params.append(f"{company_code.upper()}%")

    if years:
        # Match ANY of the years mentioned
        year_conds = ["name LIKE ?" for _ in years]
        conditions.append(f"({' OR '.join(year_conds)})")
        params.extend(f"%{y}%" for y in years)

    if quarter:
        # Quarter must appear (e.g., Q2 in 3M_2023Q2_10Q.pdf)
        conditions.append("UPPER(name) LIKE ?")
        params.append(f"%{quarter.upper()}%")

    where = " AND ".join(conditions)
    sql = f"SELECT id FROM files WHERE {where}"

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    file_ids = [row[0] for row in rows]

    logger.info(
        "File filter: company=%s years=%s quarter=%s → %d files matched",
        company_code,
        years,
        quarter,
        len(file_ids),
    )
    return file_ids


def get_file_ids_by_names(
    db_path: str,
    file_names: List[str],
) -> List[str]:
    """Find file_ids for exact file name matches.

    Args:
        db_path: Path to SQLite database.
        file_names: List of file names (e.g., ["PEPSICO_2021_10K.pdf"]).

    Returns:
        List of matching file_ids.
    """
    if not file_names:
        return []

    placeholders = ",".join("?" * len(file_names))
    sql = f"SELECT id FROM files WHERE name IN ({placeholders})"

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(sql, file_names).fetchall()
    finally:
        conn.close()

    file_ids = [row[0] for row in rows]

    logger.info(
        "File filter by names: %d names → %d file_ids matched",
        len(file_names),
        len(file_ids),
    )
    return file_ids


def get_all_file_names(db_path: str) -> List[str]:
    """Get all file names from the database.

    Returns:
        Sorted list of all file names.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM files ORDER BY name").fetchall()
    finally:
        conn.close()

    return [row[0] for row in rows]
