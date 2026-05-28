"""`_write_repo_records` must downgrade EmptyRepoError to a debug-level skip.

Template / placeholder repos with no commits surface as `EmptyRepoError` from
the clone layer (HEAD doesn't resolve). They're expected, not failures — the
consumer counts them as a skip without emitting a WARNING. Other CloneError
subclasses still warn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from github_twin.config import IngestCfg
from github_twin.ingest import commits as commits_mod
from github_twin.ingest.clone import CloneError, EmptyRepoError
from github_twin.store.db import open_db
from tests.conftest import seed_target


@pytest.fixture
def conn(tmp_path: Path):
    db = open_db(tmp_path / "test.sqlite", embed_dim=4)
    seed_target(db)
    yield db
    db.close()


def _make_records(error: BaseException) -> commits_mod._RepoRecords:
    return commits_mod._RepoRecords(repo_full="org/test-demo", head_sha="", error=error)


def test_empty_repo_skips_without_warning(conn, caplog):
    stats = commits_mod.CommitStats()
    records = _make_records(EmptyRepoError("org/test-demo: repository has no commits"))

    with caplog.at_level(logging.WARNING, logger="github_twin.ingest.commits"):
        commits_mod._write_repo_records(
            conn=conn,
            gh=None,  # never touched on the error branch
            cfg=IngestCfg(),
            target_id=1,
            records=records,
            stats=stats,
            resolve_author_login=False,
        )

    assert stats.skipped == 1
    assert caplog.records == []


def test_other_clone_errors_still_warn(conn, caplog):
    stats = commits_mod.CommitStats()
    records = _make_records(CloneError("git fetch failed (128): fatal: remote hung up"))

    with caplog.at_level(logging.WARNING, logger="github_twin.ingest.commits"):
        commits_mod._write_repo_records(
            conn=conn,
            gh=None,
            cfg=IngestCfg(),
            target_id=1,
            records=records,
            stats=stats,
            resolve_author_login=False,
        )

    assert stats.skipped == 1
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "walk org/test-demo failed" in warns[0].getMessage()
