# Demo assets

This directory holds the visual assets embedded in the top of the project
`README.md`. Two artifacts are expected:

- `demo.gif` — 30-second terminal recording of an end-to-end query path.
- `claude-code-screenshot.png` — Claude Code citing a github-twin hit.

Both are captured manually. The recipes below produce results that match
the framing in the README.

## `demo.gif` — terminal recording

Goal: show a stranger that retrieval works against a real corpus in
under 30 seconds. The cast must terminate cleanly so the GIF loops.

### Recipe

```sh
# 1. Pre-warm a target DB outside the recording so ingest noise isn't
#    captured. Pick a directory holding ~50 commits from a real repo.
export GT_PATHS__DATA_DIR=~/github-twin-demo

uvx github-twin auth login    # one-time
uvx github-twin init --kind user
uvx github-twin sync --since 2025-01-01

# 2. Record the cast. Asciinema captures the stdout stream as a .cast file.
asciinema rec docs/assets/demo.cast \
  --cols 100 --rows 28 --idle-time-limit 1.5

# Inside the recording session, run two short commands:
#   gt stats
#   gt eval search evals/queries/default.yaml
# Then exit (Ctrl+D).

# 3. Convert the .cast to a looping GIF with agg.
agg --font-size 14 --theme monokai docs/assets/demo.cast docs/assets/demo.gif
```

### Constraints

- **Keep the cast ≤ 30 seconds**; pause the runner with `--idle-time-limit`
  rather than editing the cast afterward.
- **No secrets on screen** — `gt --verbose` paths log redacted tokens, but
  the cast also captures shell history; double-check the prompt before
  starting.
- **Final width ≤ 100 cols** so the gif renders on the PyPI page without
  side-scroll. PyPI's project description renders Markdown but doesn't
  let users widen the column.

## `claude-code-screenshot.png` — Claude Code in the loop

Goal: show the *user-facing* moment when Claude cites a past commit /
review against new code. A "screenshot of a terminal" reads as
"another CLI tool" to most reviewers; a screenshot of Claude Code
citing a github-twin chunk reads as "this plugs in here."

### Recipe

1. Open Claude Code in a directory that has the github-twin MCP server
   wired in (`/mcp` should list `github-twin` as connected).
2. Ask: "Review this function in the style of my past review comments:
   `<paste a small diff>`." Claude will call `find_review_comments` or
   `find_applicable_rules` and quote the hit.
3. Take a screenshot of the response panel showing both:
   - the called tool (`find_review_comments`)
   - the quoted past comment with `repo`, `pr_number`, and `author` fields
4. Redact any internal repo/PR identifiers before committing. macOS
   Preview can blur regions; on Linux, `gimp` or `pinta` does the same.
5. Save as `docs/assets/claude-code-screenshot.png`.

### Constraints

- **Square-ish aspect, ≤ 1200 px wide** — keeps the README hero compact.
- **No private code in the diff**; use a snippet from a public repo
  (e.g. `modelcontextprotocol/servers`) so the screenshot is shareable.

## After capturing

Both files are referenced from the top of `README.md`. Once they
exist, no further wiring is needed — the README's existing image tags
will pick them up.
