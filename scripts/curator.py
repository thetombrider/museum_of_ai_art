#!/usr/bin/env python3
"""
Museum of AI Art — daily curator.

Fetches a random "on this day" historical event from Wikipedia, asks Claude Haiku
to write a pretentious title and artist statement plus a Flux image prompt, then
submits the prompt to fal.ai's Flux model. The resulting painting is committed
to the repo alongside its metadata; the git repo itself is the database.

Designed to be idempotent: re-running for the same day is a no-op.

Environment variables:
    ANTHROPIC_API_KEY   required
    FAL_KEY             required
    MUSEUM_DATE         optional, override "today" as YYYY-MM-DD (for testing/backfill)
    MUSEUM_REPO_ROOT    optional, defaults to parent of this file's parent
    MUSEUM_DRY_RUN      if "1", skip fal.ai and write a stub painting instead
"""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import requests


# ----------------------------- configuration --------------------------------

WIKI_BASE = "https://en.wikipedia.org/api/rest_v1/feed/onthisday"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("MUSEUM_HAIKU_MODEL", "claude-haiku-4-5")
ANTHROPIC_VERSION = "2023-06-01"

FAL_MODEL = os.environ.get("MUSEUM_FAL_MODEL", "fal-ai/flux/dev")
FAL_IMAGE_SIZE = os.environ.get("MUSEUM_FAL_SIZE", "landscape_4_3")

USER_AGENT = "MuseumOfAIArt/1.0 (https://github.com/; curator bot)"

REPO_ROOT = Path(os.environ.get("MUSEUM_REPO_ROOT", Path(__file__).resolve().parent.parent))
GALLERY_DIR = REPO_ROOT / "gallery"
EXHIBITS_DIR = GALLERY_DIR / "exhibits"
INDEX_PATH = GALLERY_DIR / "exhibit.json"


# ----------------------------- wikipedia ------------------------------------


def fetch_onthisday(date: dt.date) -> list[dict[str, Any]]:
    """Return the curated `selected` events for a date, falling back to `events`."""
    url_selected = f"{WIKI_BASE}/selected/{date.month:02d}/{date.day:02d}"
    r = requests.get(url_selected, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    data = r.json()
    pool = data.get("selected") or []
    if pool:
        return pool
    # Fallback to the full events list for that date.
    url_events = f"{WIKI_BASE}/events/{date.month:02d}/{date.day:02d}"
    r2 = requests.get(url_events, headers={"User-Agent": USER_AGENT}, timeout=30)
    r2.raise_for_status()
    return r2.json().get("events") or []


def pick_event(date: dt.date) -> dict[str, Any]:
    """Pick one event at random. Deduplicates by year so two events from the same
    year don't both reference the same article repeatedly."""
    pool = fetch_onthisday(date)
    if not pool:
        raise RuntimeError(f"No on-this-day events found for {date.isoformat()}")
    # Prefer a deterministic seed per day so re-runs of the same day pick the
    # same event (important for backfills and idempotency).
    rng = random.Random(f"museum-{date.isoformat()}")
    return rng.choice(pool)


def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    """Trim an event down to the fields we actually want to feed the LLM
    and store in the exhibit's meta.json."""
    pages = event.get("pages") or []
    title = ""
    extract = ""
    thumbnail_url = None
    wikipedia_url = None
    if pages:
        first = pages[0]
        title = (
            first.get("titles", {}).get("normalized")
            or first.get("title")
            or ""
        )
        extract = (first.get("extract") or "").strip()
        thumb = first.get("thumbnail") or first.get("originalimage") or {}
        thumbnail_url = thumb.get("source")
        canonical = (
            first.get("content_urls", {}).get("desktop", {}).get("page")
            or first.get("titles", {}).get("canonical")
        )
        if canonical and canonical.startswith("http"):
            wikipedia_url = canonical
        elif canonical:
            wikipedia_url = f"https://en.wikipedia.org/wiki/{canonical}"
    return {
        "year": event.get("year"),
        "text": (event.get("text") or "").strip(),
        "page_title": title.strip(),
        "page_extract": extract,
        "thumbnail_url": thumbnail_url,
        "wikipedia_url": wikipedia_url,
    }


# ----------------------------- claude haiku ---------------------------------


HAIKU_SYSTEM = textwrap.dedent(
    """
    You are the resident curator of the Museum of AI Art, a small but extremely
    pretentious institution that hangs a single new painting every day. Each
    painting is an AI-generated oil-on-canvas interpretation of a real historical
    event that happened on this day.

    Your job is to produce, for the event the user provides:
      1. "title"       — a museum-card title. Short (3-9 words), evocative,
                          art-historical in tone, no quotes, no colons. Think
                          Hopper, Richter, Hammershøi, but obscure. A small
                          number of museum-card titles include the year in
                          Roman numerals.
      2. "medium"      — a one-line faux medium statement, e.g.
                          "Oil on linen, 2026" or "Egg tempera and gesso on
                          birch panel". Just one line.
      3. "artist_statement" — a 2-4 sentence artist statement in the voice of a
                              contemporary figurative painter who is deeply
                              moved by this historical event. Be sincere and
                              slightly melancholy. Avoid academic jargon
                              and avoid referencing the event by its Wikipedia
                              name; respond to its emotional register instead.
      4. "image_prompt" — a vivid, concrete image prompt for a text-to-image
                          model. ~60-120 words. Single paragraph. The painting
                          should look like a serious oil painting: painterly
                          brushwork, controlled palette, single dominant light
                          source, museum-grade composition. Do NOT mention the
                          event by name. Do NOT name real people. Do NOT use
                          the words "AI", "generated", "digital" or
                          "painting of". Describe a specific scene with
                          specific objects, light, weather, and texture.

    Return strict JSON with exactly these four keys: title, medium,
    artist_statement, image_prompt. No prose, no markdown, no code fences.
    """
).strip()


def call_haiku(event: dict[str, Any]) -> dict[str, str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    user_payload = json.dumps(event, ensure_ascii=False, indent=2)
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": HAIKU_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Today's historical event:\n\n" + user_payload + "\n\n"
                    "Return JSON only."
                ),
            }
        ],
    }
    r = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        json=body,
        timeout=120,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Anthropic API error {r.status_code}: {r.text[:500]}")
    data = r.json()
    text = "".join(
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    )
    return parse_haiku_json(text)


def parse_haiku_json(text: str) -> dict[str, str]:
    """The model is told to return strict JSON, but be defensive."""
    text = text.strip()
    # Strip a single leading/trailing code fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Last-ditch: grab the first {...} block.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise RuntimeError(f"Haiku did not return JSON: {text[:300]!r}")
        obj = json.loads(m.group(0))
    required = {"title", "medium", "artist_statement", "image_prompt"}
    missing = required - set(obj)
    if missing:
        raise RuntimeError(f"Haiku JSON missing keys {missing}: {obj}")
    return {k: str(obj[k]).strip() for k in required}


# ----------------------------- fal.ai flux ----------------------------------


def generate_image(prompt: str) -> bytes:
    """Submit to fal.ai Flux via fal-client and return the image bytes."""
    if os.environ.get("MUSEUM_DRY_RUN") == "1":
        # Tiny stub PNG so the rest of the pipeline can be tested offline.
        return _stub_png_bytes()

    try:
        import fal_client  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "fal-client is not installed; run `pip install -r requirements.txt`"
        ) from e

    # fal_client.subscribe blocks until the request completes and returns the
    # full result dict, which for Flux contains an `images` list with URLs.
    result = fal_client.subscribe(
        FAL_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": FAL_IMAGE_SIZE,
            "num_images": 1,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "enable_safety_checker": True,
        },
        with_logs=False,
    )
    images = result.get("images") or []
    if not images:
        raise RuntimeError(f"fal.ai returned no images: {result}")
    url = images[0].get("url")
    if not url:
        raise RuntimeError(f"fal.ai image missing url: {images[0]}")
    img_r = requests.get(url, timeout=120)
    img_r.raise_for_status()
    return img_r.content


def _stub_png_bytes() -> bytes:
    """Minimal 1x1 PNG used only in dry-run / offline testing."""
    import base64
    b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    return base64.b64decode(b64)


# ----------------------------- persistence ----------------------------------


def exhibit_dir_for(date: dt.date) -> Path:
    return EXHIBITS_DIR / date.isoformat()


def meta_path_for(date: dt.date) -> Path:
    return exhibit_dir_for(date) / "meta.json"


def already_curated(date: dt.date) -> bool:
    return meta_path_for(date).exists()


def write_artifact(date: dt.date, event: dict[str, Any], curator: dict[str, str], image: bytes) -> dict[str, Any]:
    d = exhibit_dir_for(date)
    d.mkdir(parents=True, exist_ok=True)
    img_path = d / "painting.jpg"
    img_path.write_bytes(image)

    meta = {
        "date": date.isoformat(),
        "year": event.get("year"),
        "event": {
            "text": event.get("text"),
            "page_title": event.get("page_title"),
            "page_extract": event.get("page_extract"),
            "thumbnail_url": event.get("thumbnail_url"),
            "wikipedia_url": event.get("wikipedia_url"),
        },
        "title": curator["title"],
        "medium": curator["medium"],
        "artist_statement": curator["artist_statement"],
        "image_prompt": curator["image_prompt"],
        "image": "painting.jpg",
        "model": {
            "curator": ANTHROPIC_MODEL,
            "image": FAL_MODEL,
        },
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta_p = d / "meta.json"
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def update_index() -> list[dict[str, Any]]:
    """Rebuild exhibit.json from the filesystem. Sort newest first."""
    if not EXHIBITS_DIR.exists():
        EXHIBITS_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for meta_path in sorted(EXHIBITS_DIR.glob("*/meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        entries.append(
            {
                "date": meta.get("date"),
                "year": meta.get("year"),
                "title": meta.get("title"),
                "medium": meta.get("medium"),
                "excerpt": _excerpt(meta.get("artist_statement", ""), 220),
                "url": f"exhibits/{meta.get('date')}/",
                "thumbnail": f"exhibits/{meta.get('date')}/painting.jpg",
            }
        )
    INDEX_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    return entries


def _excerpt(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


# ----------------------------- main -----------------------------------------


def main() -> int:
    date_str = os.environ.get("MUSEUM_DATE")
    date = dt.date.fromisoformat(date_str) if date_str else dt.date.today()

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    EXHIBITS_DIR.mkdir(parents=True, exist_ok=True)

    if already_curated(date):
        print(f"[museum] exhibit for {date.isoformat()} already exists; refreshing index only.")
        update_index()
        return 0

    print(f"[museum] curating {date.isoformat()} …")
    raw_event = pick_event(date)
    event = event_summary(raw_event)
    print(f"[museum] event: {event['year']} — {event['text'][:100]}…")

    curator = call_haiku(event)
    print(f"[museum] title: {curator['title']}")
    print(f"[museum] image prompt: {curator['image_prompt'][:100]}…")

    image = generate_image(curator["image_prompt"])
    meta = write_artifact(date, event, curator, image)
    update_index()
    print(f"[museum] wrote {meta_path_for(date)} ({len(image)} bytes image)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
