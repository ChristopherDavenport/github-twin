"""github-twin: personal RAG over your GitHub history.

`__version__` is written by `hatch-vcs` into `_version.py` at build time
(see `[tool.hatch.build.hooks.vcs]` in pyproject.toml). When running
from a source checkout that hasn't been built, fall back to reading the
installed dist metadata, then to a sentinel for the very-development
case.
"""

from __future__ import annotations

__version__: str

try:
    from github_twin._version import __version__
except ImportError:  # not built yet — fall back to installed metadata
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _pkg_version

        try:
            __version__ = _pkg_version("github-twin")
        except PackageNotFoundError:
            __version__ = "0.0.0+unknown"
    except ImportError:  # pragma: no cover - very-old python
        __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
