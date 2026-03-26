"""
Extract metadata (company, fiscal year) from a query for file-level filtering.

Two modes:
  - Rule-based: regex extraction of company name + fiscal year
  - LLM-based: ask LLM to select the most relevant file(s) from a list
"""

import json
import logging
import re
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Year extraction patterns (ordered by specificity)
# ---------------------------------------------------------------------------
_YEAR_PATTERNS = [
    # "Q2 of FY2023", "Q2 FY2023", "Q2 of FY 2023"
    re.compile(r"Q([1-4])\s+(?:of\s+)?FY\s*(\d{4})", re.IGNORECASE),
    # "FY2023", "FY 2023", "fiscal year 2023"
    re.compile(r"(?:FY|fiscal\s+year)\s*(\d{4})", re.IGNORECASE),
    # Standalone 4-digit year preceded by context words
    re.compile(r"(?:in|for|during|as\s+of|from|between)\s+(\d{4})", re.IGNORECASE),
]

# Known aliases: benchmark company name → file code
# Built from financebench_open_source.jsonl analysis
_HARDCODED_ALIASES: Dict[str, str] = {
    # Cases where company name differs significantly from file code
    "j&j": "JOHNSON_JOHNSON",
    "jnj": "JOHNSON_JOHNSON",
    "johnson & johnson": "JOHNSON_JOHNSON",
    "johnson and johnson": "JOHNSON_JOHNSON",
    "pg&e": "PG_E",
    "pacific gas": "PG_E",
    "coca-cola": "COCACOLA",
    "coca cola": "COCACOLA",
    "american express": "AMERICANEXPRESS",
    "amex": "AMERICANEXPRESS",
    "american water works": "AMERICANWATERWORKS",
    "american water": "AMERICANWATERWORKS",
    "activision blizzard": "ACTIVISIONBLIZZARD",
    "activision": "ACTIVISIONBLIZZARD",
    "aes corporation": "AES",
    "boston properties": "BOSTONPROPERTIES",
    "best buy": "BESTBUY",
    "cvs health": "CVSHEALTH",
    "cvs": "CVSHEALTH",
    "general mills": "GENERALMILLS",
    "kraft heinz": "KRAFTHEINZ",
    "lockheed martin": "LOCKHEEDMARTIN",
    "lockheed": "LOCKHEEDMARTIN",
    "mcdonald's": "MCDONALDS",
    "mcdonalds": "MCDONALDS",
    "mgm resorts": "MGMRESORTS",
    "mgm": "MGMRESORTS",
    "foot locker": "FOOTLOCKER",
    "jpmorgan": "JPMORGAN",
    "jp morgan": "JPMORGAN",
    "jpmorgan chase": "JPMORGAN",
    "ulta beauty": "ULTABEAUTY",
    "ulta": "ULTABEAUTY",
    "salesforce": "SALESFORCE",
}


def build_company_map(db_path: str) -> Dict[str, str]:
    """Build a mapping from normalized company name → file company code.

    Reads all file names from the SQLite files table, extracts company codes,
    and creates a case-insensitive lookup map including hardcoded aliases.

    Returns:
        Dict mapping lowercase name/alias → uppercase company code.
        Example: {"pepsico": "PEPSICO", "j&j": "JOHNSON_JOHNSON", ...}
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT name FROM files").fetchall()
    finally:
        conn.close()

    # Extract company codes from file names
    company_codes = set()
    for (name,) in rows:
        # Strip .pdf extension
        base = name.rsplit(".", 1)[0] if "." in name else name
        # Extract company part: everything before the first _YYYY pattern
        m = re.match(r"^(.+?)_\d{4}", base)
        if m:
            company_codes.add(m.group(1).upper())
        else:
            # Files like MCDONALDS_8K_dated-2023-02-13.pdf
            m2 = re.match(r"^(.+?)_8K", base)
            if m2:
                company_codes.add(m2.group(1).upper())

    # Build map: lowercase → uppercase code
    cmap: Dict[str, str] = {}

    for code in company_codes:
        # Direct lowercase match
        cmap[code.lower()] = code
        # Without underscores: "JOHNSON_JOHNSON" → "johnsonjohnson"
        cmap[code.replace("_", "").lower()] = code
        # With spaces: "JOHNSON_JOHNSON" → "johnson johnson"
        cmap[code.replace("_", " ").lower()] = code

    # Add hardcoded aliases
    for alias, code in _HARDCODED_ALIASES.items():
        cmap[alias.lower()] = code

    logger.info(
        "Built company map: %d codes, %d aliases (total %d entries)",
        len(company_codes),
        len(_HARDCODED_ALIASES),
        len(cmap),
    )
    return cmap


def extract_metadata_rules(
    query: str,
    company_map: Dict[str, str],
) -> Dict[str, Optional[str]]:
    """Extract company code and fiscal year from query using regex rules.

    Args:
        query: Natural language query.
        company_map: Output of build_company_map().

    Returns:
        {"company_code": "PEPSICO" or None, "year": "2021" or None, "quarter": "Q2" or None}
    """
    result: Dict[str, Any] = {
        "company_code": None,
        "years": [],  # all years mentioned in query
        "quarter": None,
    }

    # --- Year extraction ---
    # Collect ALL years mentioned in the query, then pick the latest.
    # Rationale: "FY2018 - FY2020 3 year average" needs the FY2020 10K
    # because the most recent filing contains historical data.
    all_years: list[str] = []
    for pattern in _YEAR_PATTERNS:
        for m in pattern.finditer(query):
            groups = m.groups()
            if len(groups) == 2:
                # Q-pattern: (quarter, year)
                result["quarter"] = f"Q{groups[0]}"
                all_years.append(groups[1])
            else:
                all_years.append(groups[0])
    if all_years:
        result["years"] = sorted(set(all_years))  # all unique years, sorted

    # --- Company extraction ---
    # Try longest match first (greedy matching)
    query_lower = query.lower()

    best_match = None
    best_match_len = 0

    for alias, code in company_map.items():
        # Check if alias appears in query (word boundary aware)
        # Use word boundary for short aliases to avoid false positives
        if len(alias) <= 3:
            pattern = r"(?<![a-zA-Z])" + re.escape(alias) + r"(?![a-zA-Z])"
            if re.search(pattern, query_lower):
                if len(alias) > best_match_len:
                    best_match = code
                    best_match_len = len(alias)
        else:
            if alias in query_lower:
                if len(alias) > best_match_len:
                    best_match = code
                    best_match_len = len(alias)

    result["company_code"] = best_match

    logger.info(
        "Rule-based metadata extraction: company=%s, years=%s, quarter=%s (query=%.60s...)",
        result["company_code"],
        result["years"],
        result["quarter"],
        query,
    )
    return result


# ---------------------------------------------------------------------------
# LLM-based file selection
# ---------------------------------------------------------------------------

_FILE_SELECT_SYSTEM = (
    "You are a financial document routing system. "
    "Given a question about a company's financial data, select the 1-3 most relevant "
    "document(s) from the file list that would contain the answer.\n\n"
    "Consider:\n"
    "- Company name in the question vs file name\n"
    "- Fiscal year mentioned (FY2021 → files with 2021)\n"
    "- Document type: 10-K for annual reports, 10-Q for quarterly, 8-K for events/earnings\n"
    "- If the question asks about a specific quarter (Q2), prefer 10-Q files for that quarter\n"
    "- If ambiguous, include both 10-K and relevant 8-K/10-Q files\n\n"
    "Return ONLY a JSON array of file names. Example: [\"PEPSICO_2021_10K.pdf\"]\n"
    "If you cannot determine the correct file, return an empty array: []"
)


async def extract_file_llm(
    query: str,
    file_names: List[str],
    llm_client: Any,
    model: str = "gemini-2.5-flash",
) -> Optional[List[str]]:
    """Ask LLM to select the most relevant file(s) for a query.

    Args:
        query: Natural language query.
        file_names: All available file names (e.g., ["PEPSICO_2021_10K.pdf", ...]).
        llm_client: google.genai.Client instance.
        model: Model name for generation.

    Returns:
        List of selected file names, or None on failure.
    """
    if not llm_client:
        logger.warning("No LLM client for file selection — skipping")
        return None

    # Format file list compactly
    file_list_str = "\n".join(sorted(file_names))
    user_prompt = f"Question: {query}\n\nFile list:\n{file_list_str}"

    try:
        from google.genai import types

        response = await llm_client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_FILE_SELECT_SYSTEM,
                max_output_tokens=2048,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=1024),
            ),
        )

        # Extract text from response (thinking models may have empty .text)
        try:
            text = (response.text or "").strip()
        except Exception:
            text = ""

        if not text:
            # Log full response for debugging
            cand = response.candidates[0] if response.candidates else None
            logger.warning(
                "LLM file selection got empty text. finish_reason=%s, parts=%s, prompt_feedback=%s",
                getattr(cand, "finish_reason", None) if cand else "no_candidates",
                len(getattr(getattr(cand, "content", None), "parts", None) or []) if cand else 0,
                getattr(response, "prompt_feedback", None),
            )
        # Also check all parts for non-thought text
        if not text and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", None) or []
            for part in parts:
                if not getattr(part, "thought", False) and getattr(part, "text", None):
                    text = part.text.strip()
                    break
        logger.debug("LLM file selection raw response: %r", text[:300] if text else "(empty)")
        # Extract JSON array from response (may contain markdown fences)
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            logger.warning("LLM file selection returned no JSON array: %r", text[:200])
            return None
        selected = json.loads(json_match.group())

        if isinstance(selected, list):
            # Validate: only keep names that exist in file_names
            valid = [f for f in selected if f in file_names]
            if valid:
                logger.info("LLM file selection: %s (query=%.60s...)", valid, query)
                return valid
            else:
                logger.warning(
                    "LLM returned no valid files: %r (query=%.60s...)", selected, query
                )
                return None

        logger.warning("LLM file selection returned non-list: %r", text)
        return None

    except Exception as e:
        logger.warning(
            "LLM file selection failed (type=%s): %s — skipping filter",
            type(e).__name__,
            e,
        )
        return None
