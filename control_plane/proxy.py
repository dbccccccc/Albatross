from __future__ import annotations

from typing import Iterable, Optional

import aiohttp
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def _filtered_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers:
        if k.lower() in HOP_BY_HOP_HEADERS:
            continue
        out[k] = v
    return out


async def proxy_http(
    *,
    client: aiohttp.ClientSession,
    upstream_base: str,
    path: str,
    request: Request,
    timeout_s: float = 600.0,
) -> Response:
    upstream_url = upstream_base.rstrip("/") + "/" + path.lstrip("/")
    method = request.method.upper()

    body = await request.body()
    params = list(request.query_params.multi_items())
    headers = _filtered_headers(request.headers.items())

    async with client.request(
        method,
        upstream_url,
        params=params,
        data=body if body else None,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout_s),
    ) as resp:
        resp_headers = _filtered_headers(resp.headers.items())
        content_type = resp_headers.get("content-type")

        # Stream if upstream is SSE or chunked.
        if content_type and "text/event-stream" in content_type.lower():
            async def _iter():
                async for chunk in resp.content.iter_chunked(4096):
                    yield chunk

            return StreamingResponse(
                _iter(),
                status_code=resp.status,
                headers=resp_headers,
                media_type=content_type,
            )

        data = await resp.read()
        return Response(content=data, status_code=resp.status, headers=resp_headers, media_type=content_type)

