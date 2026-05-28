"""Tests for org repo enumeration (`ingest/repos.py`).

Covers the include/exclude matcher and the shape of objects yielded by
`enumerate_org_repos`. The GitHub client is faked out — we only care that
pagination output is filtered and projected correctly.
"""

from __future__ import annotations

from typing import Any

import pytest

from github_twin.ingest.repos import (
    enumerate_org_repos,
    matches_any,
    repo_passes_filters,
)


class FakeGH:
    def __init__(self, pages: list[dict[str, Any]]):
        self._pages = pages

    def paginate(self, path: str, *, params: dict[str, Any] | None = None):
        # Smoke-check that the caller is hitting the org endpoint with the
        # expected params; if these drift, refresh this fixture too.
        assert path.startswith("/orgs/")
        assert params is not None
        assert params["per_page"] == 100
        assert params["type"] == "all"
        yield from self._pages


def _repo(name: str, **over: Any) -> dict[str, Any]:
    base = {
        "full_name": name,
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "visibility": "public",
        "fork": False,
        "size": 42,
    }
    base.update(over)
    return base


def test_matches_any():
    assert matches_any("org/foo", ["org/*"])
    assert not matches_any("org/foo", ["other/*"])
    assert matches_any("org/foo-bar", ["org/foo-*"])


@pytest.mark.parametrize(
    "name,include,exclude,expected",
    [
        ("org/a", [], [], True),  # no filters → pass
        ("org/a", ["org/*"], [], True),  # include match
        ("org/a", ["other/*"], [], False),  # include miss
        ("org/a", [], ["org/a"], False),  # explicit exclude
        ("org/a", [], ["org/*"], False),  # exclude glob
        ("org/a", ["org/*"], ["org/a"], False),  # exclude wins over include
    ],
)
def test_repo_passes_filters(name, include, exclude, expected):
    assert repo_passes_filters(name, include=include, exclude=exclude) is expected


def test_enumerate_org_repos_yields_shaped_dicts():
    gh = FakeGH(
        [
            _repo("org/a", size=10),
            _repo("org/c", fork=True, size=30, visibility="private"),
        ]
    )
    out = list(enumerate_org_repos(gh, "org"))
    assert [r["full_name"] for r in out] == ["org/a", "org/c"]
    assert out[0] == {
        "full_name": "org/a",
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "visibility": "public",
        "fork": False,
        "size_kb": 10,
    }
    assert out[1]["fork"] is True
    assert out[1]["visibility"] == "private"


def test_enumerate_org_repos_skips_archived_by_default():
    gh = FakeGH(
        [
            _repo("org/active"),
            _repo("org/archived", archived=True),
            _repo("org/internal-archived", archived=True, visibility="internal"),
        ]
    )
    out = list(enumerate_org_repos(gh, "org"))
    assert [r["full_name"] for r in out] == ["org/active"]


def test_enumerate_org_repos_keeps_archived_when_opted_in():
    gh = FakeGH(
        [
            _repo("org/active"),
            _repo("org/archived", archived=True),
            _repo("org/internal-archived", archived=True, visibility="internal"),
        ]
    )
    out = list(enumerate_org_repos(gh, "org", include_archived=True))
    assert [r["full_name"] for r in out] == [
        "org/active",
        "org/archived",
        "org/internal-archived",
    ]
    assert out[1]["archived"] is True
    assert out[2]["visibility"] == "internal"


def test_enumerate_org_repos_passes_through_missing_visibility():
    """Older GHE responses may omit `visibility`; we should store NULL, not raise."""
    raw = {
        "full_name": "org/legacy",
        "default_branch": "main",
        "pushed_at": "2024-01-01T00:00:00Z",
        "archived": False,
        "fork": False,
        "size": 1,
    }
    gh = FakeGH([raw])
    out = list(enumerate_org_repos(gh, "org"))
    assert out[0]["visibility"] is None


def test_enumerate_org_repos_applies_include():
    gh = FakeGH([_repo("org/keep"), _repo("org/drop")])
    out = list(enumerate_org_repos(gh, "org", include=["org/keep"]))
    assert [r["full_name"] for r in out] == ["org/keep"]


def test_enumerate_org_repos_applies_exclude():
    gh = FakeGH([_repo("org/keep"), _repo("org/drop")])
    out = list(enumerate_org_repos(gh, "org", exclude=["org/drop"]))
    assert [r["full_name"] for r in out] == ["org/keep"]


def test_enumerate_org_repos_skips_items_without_full_name():
    """Defensive: GitHub has historically returned shells in rare error modes."""
    gh = FakeGH([{"name": "no full_name"}, _repo("org/real")])
    out = list(enumerate_org_repos(gh, "org"))
    assert [r["full_name"] for r in out] == ["org/real"]
