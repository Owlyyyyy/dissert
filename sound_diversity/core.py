from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs without logging secrets or overriding the environment."""
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            os.environ.setdefault(key, value.strip().strip("\"'"))


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(c for c in value if not unicodedata.combining(c)).casefold()
    value = value.replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def clean_discogs_artist(value: str) -> str:
    """Remove Discogs-only disambiguation suffixes such as 'Artist (2)'."""
    return re.sub(r"\s+\(\d+\)\s*$", "", (value or "").strip())


def discogs_artist_credit(artists: list[dict[str, Any]] | None) -> str:
    """Build a primary artist credit from a Discogs artists array."""
    if not artists:
        return ""
    pieces: list[str] = []
    for artist in artists:
        name = clean_discogs_artist(artist.get("name") or artist.get("anv") or "")
        if not name:
            continue
        pieces.append(name)
        join = (artist.get("join") or "").strip()
        if join:
            pieces.append(join)
    return " ".join(pieces).strip()


def is_usable_artist(value: str) -> bool:
    artist = normalize(value)
    return bool(artist) and artist not in {
        "various", "various artists", "unknown", "unknown artist", "no artist",
    }


def is_discogs_release(item: dict[str, Any]) -> bool:
    """Distinguish concrete releases from masters in label-page responses."""
    return "/releases/" in (item.get("resource_url") or "")


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(p) for p in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    while True:
        try:
            tmp.replace(path)
            break
        except PermissionError:
            # Excel, antivirus, or a concurrent status reader can temporarily
            # lock a CSV on Windows. Preserve the completed temp checkpoint and
            # wait for the lock instead of terminating a multi-hour collection.
            time.sleep(1)


@dataclass(frozen=True)
class Settings:
    seed: int
    sample_size: int
    itunes_country: str
    itunes_match_threshold: float
    clap_model: str
    clap_revision: str
    clap_sample_rate: int
    discogs_user_agent: str

    @classmethod
    def load(cls, path: Path = ROOT / "configs/settings.json") -> "Settings":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class JsonCache:
    def __init__(self, root: Path):
        self.root = root

    def path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        path = self.path(namespace, key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def put(self, namespace: str, key: str, value: Any) -> None:
        path = self.path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class DiscogsClient:
    base = "https://api.discogs.com"

    def __init__(self, token: str, user_agent: str, cache: JsonCache):
        if not token:
            raise RuntimeError("DISCOGS_TOKEN is not set")
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Discogs token={token}",
            "User-Agent": user_agent,
        })

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        key = f"{path}?{json.dumps(params, sort_keys=True)}"
        cached = self.cache.get("discogs", key)
        if cached is not None:
            return cached
        attempts = 0
        while True:
            try:
                response = self.session.get(self.base + path, params=params, timeout=45)
            except requests.RequestException:
                attempts += 1
                # Network/DNS outages affect every release, so never convert them
                # into missing data. Wait up to a minute and keep trying until the
                # connection returns; this makes long unattended runs resumeless.
                time.sleep(min(2 ** min(attempts, 6), 60))
                continue
            if response.status_code == 429:
                time.sleep(int(response.headers.get("Retry-After", "60")))
                continue
            if response.status_code == 404:
                data = {"_http_status": 404, "_missing_resource": path}
                self.cache.put("discogs", key, data)
                return data
            if 500 <= response.status_code < 600:
                attempts += 1
                if attempts < 5:
                    time.sleep(min(2 ** attempts, 30))
                    continue
                data = {"_http_status": response.status_code, "_failed_resource": path}
                self.cache.put("discogs", key, data)
                return data
            response.raise_for_status()
            data = response.json()
            self.cache.put("discogs", key, data)
            remaining = int(response.headers.get("X-Discogs-Ratelimit-Remaining", "60"))
            if remaining < 2:
                time.sleep(60)
            return data

    def search_labels(self, name: str) -> list[dict[str, Any]]:
        return self.get("/database/search", {"q": name, "type": "label", "per_page": 20}).get("results", [])

    def label_releases(self, label_id: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self.get(f"/labels/{label_id}/releases", {"page": page, "per_page": 100})
            rows.extend(data.get("releases", []))
            pages = int(data.get("pagination", {}).get("pages", page))
            if page >= pages:
                return rows
            page += 1

    def release(self, release_id: int) -> dict[str, Any]:
        return self.get(f"/releases/{release_id}")


class ITunesClient:
    url = "https://itunes.apple.com/search"

    def __init__(self, country: str, threshold: float, cache: JsonCache):
        self.country = country
        self.threshold = threshold
        self.cache = cache
        self.last_lookup_status = "not_started"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SoundDiversityResearch/0.1"})

    @staticmethod
    def similarity(left: str, right: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, normalize(left), normalize(right)).ratio()

    def search_track(self, artist: str, track: str, retry_cached_403: bool = False) -> dict[str, Any] | None:
        params = {"term": f"{artist} {track}", "country": self.country, "media": "music", "entity": "song", "limit": 25}
        key = json.dumps(params, sort_keys=True)
        data = self.cache.get("itunes", key)
        if retry_cached_403 and data is not None and data.get("_lookup_status") == "http_403":
            data = None
        if data is None:
            attempts = 0
            while True:
                try:
                    response = self.session.get(self.url, params=params, timeout=30)
                except requests.RequestException:
                    attempts += 1
                    time.sleep(min(2 ** min(attempts, 6), 60))
                    continue
                if response.status_code == 403:
                    attempts += 1
                    if retry_cached_403 and attempts < 5:
                        time.sleep(min(5 * (2 ** (attempts - 1)), 60))
                        continue
                    if attempts < 3:
                        time.sleep(1)
                        continue
                    data = {"results": [], "_lookup_status": "http_403"}
                    self.cache.put("itunes", key, data)
                    break
                if response.status_code == 429:
                    time.sleep(int(response.headers.get("Retry-After", "30")))
                    continue
                if response.status_code == 404:
                    data = {"results": [], "_lookup_status": "http_404"}
                    self.cache.put("itunes", key, data)
                    break
                if 500 <= response.status_code < 600:
                    attempts += 1
                    time.sleep(min(2 ** min(attempts, 6), 60))
                    continue
                response.raise_for_status()
                data = response.json()
                self.cache.put("itunes", key, data)
                # Apple's Search API documentation recommends keeping calls to
                # roughly 20 per minute. Cached results do not incur this delay.
                time.sleep(3.1)
                break
        candidates = []
        for item in data.get("results", []):
            if not item.get("previewUrl"):
                continue
            artist_score = self.similarity(artist, item.get("artistName", ""))
            track_score = self.similarity(track, item.get("trackName", ""))
            score = 0.45 * artist_score + 0.55 * track_score
            if artist_score >= 0.70 and track_score >= 0.70 and score >= self.threshold:
                candidates.append((score, item))
        if not candidates:
            self.last_lookup_status = data.get("_lookup_status", "no_match")
            return None
        score, item = max(candidates, key=lambda pair: (pair[0], pair[1].get("trackId", 0)))
        self.last_lookup_status = "matched"
        return {**item, "matchScore": score}


def mean_pairwise_diversity(embeddings: np.ndarray) -> float | None:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) < 2:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length embedding")
    unit = embeddings / norms
    similarities = unit @ unit.T
    upper = similarities[np.triu_indices(len(unit), k=1)]
    return float(1.0 - upper.mean())


def panel_skeleton(labels: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        cohort = int(label["cohort_year"]) if label["cohort_year"] else None
        for year in range(int(label["start_year"]), int(label["end_year"]) + 1):
            treated = int(cohort is not None and year >= cohort)
            rows.append({
                "label": label["label"], "year": year, "release_count": 0,
                "cohort_year": cohort if cohort is not None else "",
                "acquirer": label["acquirer"], "big_three": label["big_three"],
                "ever_treated": int(label["ever_treated"]), "treated": treated,
                "event_time": year - cohort if cohort is not None else "",
                "sound_diversity": "", "n_clap_tracks": 0,
            })
    return rows


PANEL_FIELDS = ["label", "year", "release_count", "cohort_year", "acquirer", "big_three", "ever_treated", "treated", "event_time", "sound_diversity", "n_clap_tracks"]
