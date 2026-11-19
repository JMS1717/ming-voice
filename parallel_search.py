"""Parallel's anonymous MCP search, adapted to Ming's existing result format."""
from __future__ import annotations

import asyncio
import threading
from urllib.parse import urlsplit

import anyio
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, PaginatedRequestParams

ENDPOINT = "https://mcp.parallel.ai/mcp"
_TIMEOUT = 12
_MAX_RESPONSE_BYTES = 1024 * 1024


class _BoundedBody(httpx2.AsyncByteStream):
    def __init__(self, stream: httpx2.AsyncByteStream):
        self.stream = stream

    async def __aiter__(self):
        size = 0
        async for chunk in self.stream:
            size += len(chunk)
            if size > _MAX_RESPONSE_BYTES:
                raise ValueError("MCP response exceeds the size limit")
            yield chunk

    async def aclose(self):
        await self.stream.aclose()


async def _anonymous_request(request: httpx2.Request):
    # Don't echo any HTTP cookies issued by the hosted endpoint.
    request.headers.pop("Cookie", None)


async def _limit_response(response: httpx2.Response):
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("Unexpected MCP response encoding")
    response.stream = _BoundedBody(response.stream)


async def _watch_cancel(cancel: threading.Event, scope: anyio.CancelScope):
    while not cancel.is_set():
        await anyio.sleep(0.1)
    scope.cancel()


async def _call_search(query: str, headers: dict[str, str], cancel: threading.Event | None) -> CallToolResult:
    result = None
    try:
        with anyio.fail_after(_TIMEOUT):
            async with anyio.create_task_group() as tasks:
                if cancel is not None:
                    tasks.start_soon(_watch_cancel, cancel, tasks.cancel_scope)
                try:
                    # Use Ming's project User-Agent for every request in the MCP session.
                    async with (
                        httpx2.AsyncClient(
                            headers={**headers, "Accept-Encoding": "identity"},
                            timeout=_TIMEOUT,
                            max_redirects=0,  # The MCP transport can follow same-origin redirects.
                            event_hooks={"request": [_anonymous_request], "response": [_limit_response]},
                        ) as http,
                        streamable_http_client(ENDPOINT, http_client=http) as (read, write),
                        ClientSession(read, write) as session,
                    ):
                        await session.initialize()
                        page = await session.list_tools()
                        while not any(tool.name == "web_search" for tool in page.tools):
                            if not page.next_cursor:
                                raise ValueError("Parallel MCP does not offer web_search")
                            page = await session.list_tools(params=PaginatedRequestParams(cursor=page.next_cursor))
                        result = await session.call_tool("web_search", {"objective": query, "search_queries": [query]})
                finally:
                    tasks.cancel_scope.cancel()
    except TimeoutError:
        # Session cleanup can use the rest of the budget after a completed call.
        if result is None:
            raise
    if not isinstance(result, CallToolResult) or result.is_error:
        raise ValueError("Parallel MCP search did not succeed")
    return result


def _source_text(text: str) -> str:
    # Keep remote text in the observation instead of allowing role or tool delimiters.
    return " ".join(text.split()).replace("<", "‹").replace(">", "›")


def search(query: str, headers: dict[str, str], cancel: threading.Event | None) -> list[dict]:
    if cancel is not None and cancel.is_set():
        raise ValueError("Search cancelled")
    result = asyncio.run(_call_search(query, headers, cancel))
    payload = result.structured_content
    items = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Invalid Parallel MCP search results")
    hits = []
    for item in items:
        url = item.get("url") if isinstance(item, dict) else None
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            continue
        if "\\" in url or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
            continue
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if not parsed.hostname or parsed.username is not None:
            continue
        title, excerpts = item.get("title"), item.get("excerpts")
        title = _source_text(title)[:160] if isinstance(title, str) else ""
        body = _source_text(" ".join(e for e in excerpts if isinstance(e, str))) if isinstance(excerpts, list) else ""
        if title or body:
            hits.append({"title": title, "href": url, "body": body[:300], "source": _source_text(parsed.hostname)[:120]})
    if items and not hits:
        raise ValueError("No usable sources in Parallel MCP search results")
    return hits
