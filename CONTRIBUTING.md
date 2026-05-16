# Contributing

Thanks for your interest. github-twin is a small, opinionated project, so
the contribution loop is short.

## Dev setup

```sh
uv sync --extra st --extra faiss --dev
```

That mirrors what CI installs (`.github/workflows/ci.yml`), so anything
that passes locally will pass there.

## Checks before opening a PR

Run all four — CI runs the same set and will block on any failure.

```sh
uv run pytest -q
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy
```

Use `uv run ruff format src/ tests/` to fix formatting complaints.

## Filing changes

Open an issue first for anything non-trivial so we can agree on scope
before you write code. Bug reports and feature requests have templates
under `.github/ISSUE_TEMPLATE/`.

## Security issues

Do not file public issues for vulnerabilities. See
[SECURITY.md](SECURITY.md) for the reporting channel.

## For agentic contributors

[CLAUDE.md](CLAUDE.md) is the design reference — schema invariants,
pluggable backends, retrieval contracts, and the project layout. Read it
before proposing structural changes.
