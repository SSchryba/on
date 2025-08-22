"""Minimal test configuration (no mock data)."""

import os
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def test_client():
    # Require API_KEY env for tests (set a non-empty value if not provided)
    if not os.getenv("API_KEY"):
        os.environ["API_KEY"] = "test_key_temp"  # ephemeral for test run only
    return TestClient(app)
