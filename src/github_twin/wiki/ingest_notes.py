"""Round-trip ingest: `<vault>/scratch/*.md` → `kind='note'` artifacts.

Karpathy's "file outputs back into the wiki" loop: any markdown dropped
into the scratch folder becomes a first-class chunk in the corpus, so
the next hybrid_search / find_review_comments / find_code / etc. can
hit it like any GitHub-derived content. The embed-time prefix in
`embed/prefix.py` adds a `# note: {title or path}` header so vector
queries can land on notes by topic.

Idempotency:
- The artifact's `external_id` is the SHA-256 of file content. Editing a
  note hashes to a new id, so the old artifact disappears (because its
  file path is still present but its `external_id` doesn't match
  anything on disk) and the new one is inserted fresh.
- Deletion: any DB note whose `source_url` file doesn't exist anymore
  gets `delete_artifact`-ed, which cascades chunks + vectors.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from github_twin.store import queries as q
from github_twin.store.db import transaction
from github_twin.wiki.scan import iter_scratch_notes

Reporter = Callable[[str], None]

_HEADING_RE = re.compile(r"^#+\s+(.+)$", re.MULTILINE)


def _noop(_: str) -> None:
    return None


def _extract_title(body: str, fallback: str) -> str:
    """First markdown heading wins; otherwise fall back (usually the
    filename without extension). Stripped to a single line."""
    m = _HEADING_RE.search(body)
    if m:
        return m.group(1).strip()
    return fallback


def _chunk_markdown(text: str, *, max_chars: int) -> list[str]:
    """Split into windows of at most `max_chars`, preferring to cut at
    blank lines so paragraphs stay together. Short notes return as one
    chunk; very long ones are sliced into roughly even pieces.
    """
    text = text.strip("\n")
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    for para in paragraphs:
        if not current:
            current = para
            continue
        candidate = current + "\n\n" + para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            out.append(current)
            current = para
    if current:
        out.append(current)
    # Defensive: if a single paragraph is itself longer than max_chars,
    # hard-split it so we don't emit a giant chunk.
    really_out: list[str] = []
    for piece in out:
        if len(piece) <= max_chars:
            really_out.append(piece)
            continue
        for i in range(0, len(piece), max_chars):
            really_out.append(piece[i : i + max_chars])
    return really_out


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_notes(
    conn: sqlite3.Connection,
    *,
    scratch_dir: Path,
    target_id: int,
    note_chunk_chars: int = 1200,
    report: Reporter = _noop,
) -> dict[str, int]:
    """Sync DB note-artifacts with on-disk scratch files.

    Returns counts `{added, updated, unchanged, deleted}`.

    `target_id` is the FK for the new artifacts. Notes are vault-scoped
    rather than truly target-scoped — the FK just satisfies the schema.
    The caller picks the primary target (usually the lone user or first
    org target in the DB).
    """
    on_disk: dict[Path, tuple[str, str]] = {}
    for path in iter_scratch_notes(scratch_dir):
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        digest = _content_hash(body)
        on_disk[path] = (digest, body)

    existing = q.list_note_artifacts(conn, target_id=target_id)
    by_external: dict[str, dict[str, Any]] = {
        row["external_id"]: row for row in existing if row["external_id"]
    }
    by_source_url: dict[str, dict[str, Any]] = {
        row["source_url"]: row for row in existing if row["source_url"]
    }

    added = updated = unchanged = deleted = 0
    seen_artifact_ids: set[int] = set()

    with transaction(conn):
        for path, (digest, body) in on_disk.items():
            source_url = f"file://{path.resolve()}"
            existing_by_hash = by_external.get(digest)
            if existing_by_hash is not None:
                seen_artifact_ids.add(existing_by_hash["id"])
                if existing_by_hash["source_url"] == source_url:
                    unchanged += 1
                    continue
                # Same content, moved file: just update source_url, no re-chunk.
                conn.execute(
                    "UPDATE artifact SET source_url = ? WHERE id = ?",
                    (source_url, existing_by_hash["id"]),
                )
                updated += 1
                continue

            existing_by_path = by_source_url.get(source_url)
            title = _extract_title(body, fallback=path.stem)
            meta = {"path": str(path), "title": title}

            artifact_id = q.upsert_note_artifact(
                conn,
                target_id=target_id,
                external_id=digest,
                source_url=source_url,
                meta=meta,
            )
            seen_artifact_ids.add(artifact_id)

            # Replace any chunks under this artifact (idempotent re-ingest).
            q.delete_chunks_for_artifact(conn, artifact_id)
            for piece in _chunk_markdown(body, max_chars=note_chunk_chars):
                q.insert_chunk(
                    conn,
                    artifact_id=artifact_id,
                    kind="note",
                    text=piece,
                    context={"path": str(path), "title": title},
                    language="markdown",
                )

            if existing_by_path is not None and existing_by_path["id"] != artifact_id:
                # Content edited at the same path → old hash-keyed artifact is
                # now orphaned (no on-disk file matches its hash). Drop it
                # here and mark its id seen so the post-loop cleanup doesn't
                # double-count it as a separate deletion.
                q.delete_artifact(conn, existing_by_path["id"])
                seen_artifact_ids.add(existing_by_path["id"])
                updated += 1
            else:
                added += 1

        # Anything still in the DB whose artifact_id we didn't visit → its
        # file either no longer exists or its content changed so it's now
        # orphaned. Drop it.
        for row in existing:
            if row["id"] in seen_artifact_ids:
                continue
            q.delete_artifact(conn, row["id"])
            deleted += 1

    report(f"notes: {added} added, {updated} updated, {unchanged} unchanged, {deleted} deleted")
    return {
        "added": added,
        "updated": updated,
        "unchanged": unchanged,
        "deleted": deleted,
    }
