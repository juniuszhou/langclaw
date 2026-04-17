"""Tools for DeepClaw specialists: HTTP fetch/crawl helpers."""

from __future__ import annotations

import json
import re
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from langchain_core.tools import tool


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@tool
def fetch_url(url: str, max_chars: int = 48_000) -> str:
    """Fetch a public HTTP(S) URL and return readable text (HTML tags stripped).

    Use for crawling or summarizing a specific page. Prefer `web_search` for broad queries.
    Only http/https schemes are allowed.

    Args:
        url: Full URL to fetch.
        max_chars: Truncate body to this many characters (default 48000).

    Returns:
        Plain-ish text body or an error message.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return f"Error: unsupported URL scheme (only http/https): {url!r}"
    if not parsed.netloc:
        return f"Error: invalid URL: {url!r}"

    req = Request(
        url,
        headers={
            "User-Agent": "DeepClawBot/1.0 (+https://github.com/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=45) as resp:
            raw = resp.read()
            charset = "utf-8"
            ct = resp.headers.get_content_charset() if resp.headers else None
            if ct:
                charset = ct
            body = raw.decode(charset, errors="replace")
    except HTTPError as e:
        try:
            detail = e.read().decode(errors="replace")[:2000]
        except Exception:
            detail = str(e)
        return f"HTTP error {e.code} for {url}: {detail}"
    except URLError as e:
        return f"Error: could not fetch URL: {e.reason}"
    except Exception as e:
        return f"Error: {e}"

    text = _strip_html(body)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    meta = json.dumps({"url": url, "chars": len(text)})
    return f"[fetch metadata] {meta}\n\n{text}"
