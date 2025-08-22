"""FastAPI dependencies for session management and rate limiting."""

import asyncio
from typing import AsyncGenerator
import aiohttp
from fastapi import Depends

# Semaphore to limit concurrent API calls
API_SEMAPHORE = asyncio.Semaphore(5)


async def get_http_session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Provide a managed aiohttp session for API calls."""
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        yield session


async def get_rate_limited_session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Provide a rate-limited session for external API calls."""
    async with API_SEMAPHORE:
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            yield session
