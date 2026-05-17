# Adoption channels

Living checklist for distribution submissions and feedback surfaces. Update
this file as items land — it's the durable record of where github-twin
has been listed and what's still in flight.

## Distribution (Stream B)

In priority order. Item 1 is load-bearing; downstream catalogs ingest
from the official registry.

- [ ] **Official MCP Registry** (`registry.modelcontextprotocol.io`)
  - Requires: a release tag cut on `main` (workflow
    `.github/workflows/publish-mcp-registry.yml` auto-publishes on
    `v*` tags).
  - Manifest source: `server.json` at repo root.
  - Ownership challenge: GitHub Actions OIDC against the `io.github.<owner>/<repo>`
    namespace — already wired in the workflow.
  - Verification: after the next tag-cut, the project should appear at
    `https://registry.modelcontextprotocol.io/servers?q=github-twin`.

- [ ] **Claude Code Plugin Marketplace**
  - Existing entry in
    [`ChristopherDavenport/christopherdavenport-marketplace`](https://github.com/ChristopherDavenport/christopherdavenport-marketplace)
    is auto-installable via
    `/plugin marketplace add ChristopherDavenport/christopherdavenport-marketplace`.
  - Featured-list submission: `claude.ai/settings/plugins/submit`.
  - Requires: `.claude-plugin/plugin.json` to match the latest PyPI
    version (already synced by the release workflow).

- [ ] **awesome-mcp PRs**
  - [ ] `wong2/awesome-mcp-servers`
  - [ ] `punkpeye/awesome-mcp-servers`
  - [ ] `appcypher/awesome-mcp-servers`
  - Standard entry shape: name + 1-line description + repo URL + a tag.

- [ ] **PulseMCP listing** — auto-ingests from the official registry. After
  ~2 weeks of real usage signal, nudge the editor for featured placement
  via the contact form.

- [ ] **mcp.so** — file a submission issue with README + demo gif link.

## Feedback surfaces (Stream C)

- [ ] **GitHub Discussions enabled** on the repo. Once on, seed three
  threads (keeps the surface from feeling empty on launch):
  - [ ] "Show your retrieval hits — what's `github-twin` finding for you?"
  - [ ] "Embedding model picks — what's working?"
  - [ ] "Org-mode use cases — how are you scoping multi-tenant?"
- [x] **`gt feedback` CLI** — opens a prefilled GitHub Discussion. Lives
  in `src/github_twin/feedback.py` + `cli.py`. Tested in
  `tests/test_feedback.py`.

## Out-of-repo work tracked here so it isn't lost

- [ ] **Companion marketplace repo** — confirm
  `christopherdavenport-marketplace`'s `.claude-plugin/marketplace.json`
  has up-to-date github-twin entry. (Out of tree; touch in that repo.)
- [ ] **Demo assets** — record `docs/assets/demo.gif` and
  `docs/assets/claude-code-screenshot.png` per the recipe in
  `docs/assets/README.md`. The README already references both paths.

## Retro / acceptance

- Project visible at `registry.modelcontextprotocol.io/servers?q=github-twin`
  within 1 week of the first post-merge tag.
- Project visible at `claude.ai/.../marketplace` after submission.
- Log post-launch metrics (PyPI installs Δ, stars Δ, first issues filed)
  here once Stream B items are live.
