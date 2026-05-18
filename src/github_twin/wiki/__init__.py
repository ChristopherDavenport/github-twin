"""Markdown-vault surface on top of the SQLite RAG.

`export_wiki` materializes the corpus as an Obsidian-compatible vault
under `paths.data_dir/wiki/` (override with `--out`). `ingest_notes`
closes the round-trip: any `.md` dropped into `<vault>/scratch/` is
re-ingested as a `kind='note'` artifact on the next `gt sync`, feeding
back into hybrid retrieval.

See `CLAUDE.md` ("How to run things") and the original plan
(`/home/chris/.claude/plans/looking-at-https-github-com-tinyhumansai-smooth-starfish.md`)
for the design context.
"""

from __future__ import annotations

from github_twin.wiki.export import export_wiki, resolve_vault_root
from github_twin.wiki.ingest_notes import ingest_notes

__all__ = ["export_wiki", "ingest_notes", "resolve_vault_root"]
