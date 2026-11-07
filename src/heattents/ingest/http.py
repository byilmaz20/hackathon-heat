"""Minimal HTTP helpers (stdlib only) with polite retries for public open-data APIs."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.error
import urllib.request

USER_AGENT = "heattents-research/2.0 (open-data research client)"


def get(url: str, params: dict | None = None, retries: int = 4, timeout: int = 120, rate_limit_waits: int = 20) -> bytes:
    """GET with exponential back-off; HTTP 429 (rate limit) waits a minute and retries up to ``rate_limit_waits`` times."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    attempt = waits = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and waits < rate_limit_waits:
                waits += 1
                print(f"  rate limited, waiting 60 s ({waits}/{rate_limit_waits})", flush=True)
                time.sleep(60)
                continue
            if e.code < 500 or attempt >= retries - 1:
                raise
        except Exception:
            if attempt >= retries - 1:
                raise
        attempt += 1
        time.sleep(2 ** attempt * 3)


def get_json(url: str, params: dict | None = None, **kw):
    return json.loads(get(url, params, **kw))


def socrata(dataset: str, soql: dict, page: int = 50_000, domain: str = "data.cityofnewyork.us") -> list[dict]:
    """Page through a Socrata (SODA 2.1) query; ``soql`` must contain an $order for stable paging."""
    url = f"https://{domain}/resource/{dataset}.json"
    rows, offset = [], 0
    while True:
        chunk = get_json(url, {**soql, "$limit": page, "$offset": offset})
        rows += chunk
        if len(chunk) < page:
            return rows
        offset += page
