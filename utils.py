# utils.py - Helper functions

import os
import json
from typing import Any, Dict, Optional
from dotenv import load_dotenv


def load_env():
    """Load environment variables from a .env file if present."""
    load_dotenv()


def get_api_key(key_name: str) -> Optional[str]:
    """Return an API key from environment variables."""
    return os.getenv(key_name)


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Generic environment getter with default."""
    return os.getenv(name, default)


def get_launch_sites_from_env() -> Optional[Dict[str, Dict[str, Any]]]:
    """Parse LAUNCH_SITES_JSON env var into a dict if provided."""
    raw = os.getenv("LAUNCH_SITES_JSON")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None