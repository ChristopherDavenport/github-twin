# MCP tool reference

Auto-generated from `src/github_twin/mcp_server/server.py` via
`scripts/gen_mcp_tool_docs.py`. Do not edit by hand — re-run the script
after editing the server module.

Tools registered: **9**.

## Table of contents

- [`find_review_comments`](#find-review-comments)
- [`find_style_examples`](#find-style-examples)
- [`find_code`](#find-code)
- [`find_applicable_rules`](#find-applicable-rules)
- [`predict_review_outcome`](#predict-review-outcome)
- [`summarize_review_patterns`](#summarize-review-patterns)
- [`house_rules`](#house-rules)
- [`developer_profile`](#developer-profile)
- [`sync`](#sync)

---

## `find_review_comments`

Find past review comments on diffs similar to `diff_hunk`.

**Returns:** `list[dict[str, Any]]`

| Parameter | Type | Default |
| --- | --- | --- |
| `diff_hunk` | `str` | _required_ |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `author_login` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `scope` | `Scope` | `'all'` |
| `k` | `int` | `5` |

**Details:**

```
Args:
    diff_hunk: The new code under review (a unified-diff hunk works best).
    language: Optional language filter (e.g. 'python', 'go', 'typescript').
    repo: Optional 'owner/name' filter (org-mode).
    author_login: Optional GH login to narrow to a single reviewer.
    target: Optional target name. Unset = search across every target
        (coalesce + dedup). Each returned hit carries its target.
    scope: 'personal' resolves to the unique user-mode target;
        'project' to the unique repo-mode target; 'all' (default)
        is unscoped. Explicit kwargs win.
    k: Max results to return.
```

---

## `find_style_examples`

Find code that matches a description, for style reference.

**Returns:** `list[dict[str, Any]]`

| Parameter | Type | Default |
| --- | --- | --- |
| `query` | `str` | _required_ |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `author_login` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `scope` | `Scope` | `'all'` |
| `k` | `int` | `5` |

**Details:**

```
Args:
    query: Natural-language description of what you're trying to write.
    language: Optional language filter.
    repo: Optional 'owner/name' filter.
    author_login: Optional GH login to scope to one author.
    target: Optional target name. Unset = coalesce across every target.
    scope: 'personal' / 'project' / 'all' — see find_review_comments.
    k: Max results to return.
```

---

## `find_code`

Find source snippets at HEAD across every indexed target.

**Returns:** `list[dict[str, Any]]`

| Parameter | Type | Default |
| --- | --- | --- |
| `query` | `str` | _required_ |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `path_glob` | `str \| None` | `None` |
| `node_kind` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `k` | `int` | `5` |

**Details:**

```
Args:
    query: Natural-language description or code-shape to match.
    language: Optional language filter (e.g. 'scala', 'go').
    repo: Optional 'owner/name' filter.
    path_glob: Optional fnmatch glob applied to file path.
    node_kind: Optional tree-sitter AST node type.
    target: Optional target name. Unset = coalesce across all
        targets with cross-target dedup. Each hit carries its target.
    k: Max results to return.
```

---

## `find_applicable_rules`

Find distilled code-pattern rules that apply to a coding task.

**Returns:** `list[dict[str, Any]]`

| Parameter | Type | Default |
| --- | --- | --- |
| `query` | `str` | _required_ |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `author_login` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `k` | `int` | `5` |

**Details:**

```
Args:
    query: What you're about to write or change.
    language: Optional language filter.
    repo: Optional 'owner/name' filter.
    author_login: Optional GH login to scope to personal style.
    target: Optional target name. Unset = coalesce across all targets.
    k: Max results to return.
```

---

## `predict_review_outcome`

Predict how a candidate PR would be reviewed.

**Returns:** `dict[str, Any]`

| Parameter | Type | Default |
| --- | --- | --- |
| `diff_or_summary` | `str` | _required_ |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `author_login` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `k` | `int` | `20` |

**Details:**

```
Args:
    diff_or_summary: A diff, a PR title+body, or a free-form description.
    language: Optional language filter.
    repo: Optional 'owner/name' filter.
    author_login: Optional reviewer login.
    target: Optional target name. Unset = coalesce with dedup.
    k: How many similar past PRs to pull.
```

---

## `summarize_review_patterns`

Return distilled review rules.

**Returns:** `list[dict[str, Any]]`

| Parameter | Type | Default |
| --- | --- | --- |
| `language` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `limit` | `int` | `20` |

**Details:**

```
Args:
    language: Optional filter (e.g. 'scala', 'go').
    target: Optional target name. Unset = coalesce across all targets.
    limit: Max rules to return.
```

---

## `house_rules`

Return all distilled rules as a single Markdown block.

**Returns:** `dict[str, Any]`

| Parameter | Type | Default |
| --- | --- | --- |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `author_login` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `scope` | `Scope` | `'all'` |
| `limit` | `int` | `50` |

**Details:**

```
Args:
    language: Strongly recommended in single-language sessions.
    repo: Optional 'owner/name' filter.
    author_login: Optional GH login filter.
    target: Optional target name. Unset = coalesce across all targets.
    scope: 'personal' / 'project' / 'all' — see find_review_comments.
    limit: Max rules per source kind (review + code). Default 50.
```

---

## `developer_profile`

Synthesize a short Markdown profile of one developer's review voice.

**Returns:** `dict[str, Any]`

| Parameter | Type | Default |
| --- | --- | --- |
| `author_login` | `str \| None` | `None` |
| `language` | `str \| None` | `None` |
| `repo` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |
| `scope` | `Scope` | `'all'` |
| `n_samples` | `int` | `50` |
| `force_refresh` | `bool` | `False` |

**Details:**

```
Args:
    author_login: GH login to profile. Required in org mode; in
        user mode, omit to profile the corpus owner.
    language: Optional language filter on review-comment chunks.
    repo: Optional 'owner/name' filter.
    target: Optional target name. Unset = coalesce samples across
        every target the author appears in (with dedup).
    scope: 'personal' resolves to the user-mode target — the
        right call when you want a strictly user-mode profile
        without org-side mirrors of the same commits.
    n_samples: Number of recent review comments.
    force_refresh: Bypass the cache and re-synthesize.
```

---

## `sync`

Incremental: pull new commits + review comments and embed them.

**Returns:** `dict[str, Any]`

| Parameter | Type | Default |
| --- | --- | --- |
| `since` | `str \| None` | `None` |
| `target` | `str \| None` | `None` |

**Details:**

```
Args:
    since: ISO date floor. If omitted, uses the stored sync cursor.
    target: Optional target name. Unset = run ingest for every
        target in the DB; summarize + embed are corpus-wide.
```

