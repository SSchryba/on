"""Parsers for external public data formats (e.g., TLE)."""
from typing import List, Dict


def parse_celestrak_tle(text: str, limit: int | None = 50) -> List[Dict[str, str]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    result: List[Dict[str, str]] = []
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i:i+3]
        result.append({"name": name, "line1": l1, "line2": l2})
        if limit and len(result) >= limit:
            break
    return result
