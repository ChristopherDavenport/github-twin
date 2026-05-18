"""`gt sync` round-trip: `<vault>/scratch/*.md` → kind='note' artifacts.

Pins the add / update / delete contract, the generated-file skip, and
the integration with `export_wiki`'s `scratch/README.md` writer (so the
explainer never gets ingested as a note).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github_twin.config import Config
from github_twin.store import queries as q
from github_twin.store.db import open_db
from github_twin.wiki.export import export_wiki
from github_twin.wiki.ingest_notes import _chunk_markdown, ingest_notes
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "notes.sqlite", embed_dim=4)
    seed_target(db, kind="user", name="alice")
    yield db
    db.close()


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


# ---------- add ----------


def test_add_note_creates_artifact_and_chunks(conn, tmp_path):
    scratch = tmp_path / "scratch"
    _write(scratch / "ideas.md", "# Ideas\n\nPomegranate-grapefruit-tortoise probe.\n")
    out = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert out == {"added": 1, "updated": 0, "unchanged": 0, "deleted": 0}
    arts = q.list_note_artifacts(conn, target_id=1)
    assert len(arts) == 1
    chunks = conn.execute(
        "SELECT text FROM chunk WHERE artifact_id = ?", (arts[0]["id"],)
    ).fetchall()
    assert len(chunks) == 1
    assert "Pomegranate-grapefruit-tortoise" in chunks[0]["text"]
    # Title from first heading
    assert arts[0]["meta"]["title"] == "Ideas"


# ---------- unchanged ----------


def test_rerun_with_same_content_is_unchanged(conn, tmp_path):
    scratch = tmp_path / "scratch"
    _write(scratch / "a.md", "# A\n\nbody.\n")
    ingest_notes(conn, scratch_dir=scratch, target_id=1)
    out = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert out == {"added": 0, "updated": 0, "unchanged": 1, "deleted": 0}


# ---------- update (edit in place) ----------


def test_editing_content_swaps_artifact(conn, tmp_path):
    scratch = tmp_path / "scratch"
    note = _write(scratch / "thought.md", "# Thought\n\nfirst pass\n")
    first = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert first["added"] == 1
    orig_id = q.list_note_artifacts(conn, target_id=1)[0]["id"]
    orig_external = q.list_note_artifacts(conn, target_id=1)[0]["external_id"]

    note.write_text("# Thought\n\nsecond pass.\n", encoding="utf-8")
    second = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert second["updated"] == 1
    arts = q.list_note_artifacts(conn, target_id=1)
    assert len(arts) == 1
    # New external_id (content hash changed); old artifact id is gone.
    assert arts[0]["external_id"] != orig_external
    assert arts[0]["id"] != orig_id


# ---------- delete (file removed) ----------


def test_removing_file_deletes_artifact(conn, tmp_path):
    scratch = tmp_path / "scratch"
    note = _write(scratch / "ephemeral.md", "# Bye\n\nadios.\n")
    ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert len(q.list_note_artifacts(conn, target_id=1)) == 1
    note.unlink()
    out = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert out == {"added": 0, "updated": 0, "unchanged": 0, "deleted": 1}
    assert q.list_note_artifacts(conn, target_id=1) == []


# ---------- generated:true is skipped ----------


def test_generated_file_in_scratch_is_skipped(conn, tmp_path):
    """Defense in depth: a generated wiki page accidentally moved into
    scratch must NOT be re-ingested (the round-trip would loop)."""
    scratch = tmp_path / "scratch"
    _write(
        scratch / "rogue.md",
        "---\ngenerated: true\nsource: github-twin\n---\n\n# nope\n",
    )
    out = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    assert out == {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    assert q.list_note_artifacts(conn, target_id=1) == []


# ---------- long content splits into multiple chunks ----------


def test_long_note_splits_into_multiple_chunks(conn, tmp_path):
    scratch = tmp_path / "scratch"
    para = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 30
    body = "# Long\n\n" + "\n\n".join([para] * 4)
    _write(scratch / "long.md", body)
    ingest_notes(conn, scratch_dir=scratch, target_id=1, note_chunk_chars=500)
    arts = q.list_note_artifacts(conn, target_id=1)
    assert len(arts) == 1
    chunks = conn.execute(
        "SELECT COUNT(*) AS n FROM chunk WHERE artifact_id = ?", (arts[0]["id"],)
    ).fetchone()["n"]
    assert chunks >= 2


def test_chunk_markdown_short_returns_single_chunk():
    chunks = _chunk_markdown("# Short\n\nbody only.\n", max_chars=1200)
    assert len(chunks) == 1


def test_chunk_markdown_paragraph_split_respects_max():
    body = "\n\n".join([f"para {i} " + "x" * 100 for i in range(10)])
    chunks = _chunk_markdown(body, max_chars=300)
    assert len(chunks) >= 3
    assert all(len(c) <= 300 for c in chunks)


# ---------- integration with export_wiki ----------


def test_scratch_readme_written_by_export_is_not_ingested(conn, tmp_path):
    cfg = Config()
    cfg.paths.data_dir = tmp_path / "data"
    export_wiki(conn, cfg)
    scratch = cfg.paths.data_dir / "wiki" / "scratch"
    assert (scratch / "README.md").exists()
    out = ingest_notes(conn, scratch_dir=scratch, target_id=1)
    # README.md carries `generated: true`, so iter_scratch_notes excludes it.
    assert out == {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0}
