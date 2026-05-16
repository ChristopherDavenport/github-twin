"""On-disk cache for raw GitHub responses.

Why bother:
- Re-running ingest after a chunker change shouldn't re-hit the API.
- Cheap source of truth for debugging weird artifacts.

Layout under `data/raw/`:
  commits/<sha>.json
  commits/<sha>.diff
  reviews/<owner>__<repo>__<pr>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RawCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str, ext: str) -> Path:
        safe = key.replace("/", "__")
        d = self.root / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{safe}.{ext}"

    def write_json(self, kind: str, key: str, obj: Any) -> None:
        path = self._path(kind, key, "json")
        path.write_text(json.dumps(obj, separators=(",", ":")))

    def read_json(self, kind: str, key: str) -> Any | None:
        path = self._path(kind, key, "json")
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write_text(self, kind: str, key: str, ext: str, text: str) -> None:
        self._path(kind, key, ext).write_text(text)

    def read_text(self, kind: str, key: str, ext: str) -> str | None:
        path = self._path(kind, key, ext)
        if not path.exists():
            return None
        return path.read_text()

    def has(self, kind: str, key: str, ext: str) -> bool:
        return self._path(kind, key, ext).exists()
