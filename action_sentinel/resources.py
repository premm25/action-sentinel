"""The files Sentinel ships with: settings profiles, rule packs, the synthetic dataset and sample actions.

In the repository they live in `config/` and `examples/`. A built wheel copies them into
`action_sentinel/_bundled/` under the same relative paths (see setup.py), so `pip install action-sentinel`
works without a clone. Relative paths in settings resolve against the settings' base directory first, then
against these files, so `examples/rule_packs/core.yaml` works in both places.
`action-sentinel init` copies them into a project to edit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
# What setup.py bundles, and what `init` copies: (directory, glob) relative to the repository root.
BUNDLED = [("config", "*.yaml"), ("examples/rule_packs", "*.yaml"), ("examples/evaluation_datasets", "*.jsonl"),
           ("examples/sample_actions", "*.json*")]


def bundled_root() -> Path | None:
    """`action_sentinel/_bundled` in an installed wheel, else the source checkout (editable installs)."""
    for root in (_PACKAGE / "_bundled", _PACKAGE.parent):
        if (root / "config" / "sentinel.yaml").is_file():
            return root
    return None


def is_bundled(path: Path) -> bool:
    """True for files inside an installed wheel's `_bundled` directory (not a source checkout)."""
    try:
        path.resolve().relative_to(_PACKAGE / "_bundled")
        return True
    except ValueError:
        return False


def resolve(base: str | Path, p: str | Path) -> Path:
    """`base / p`; if that doesn't exist and Sentinel ships a file at `p`, the shipped file."""
    q = Path(p)
    if q.is_absolute():
        return q
    local = Path(base) / q
    root = bundled_root()
    if not local.exists() and root is not None and (root / q).exists():
        return root / q
    return local


def copy_bundled(dest: str | Path, force: bool = False) -> tuple[list[Path], list[Path]]:
    """Copy the shipped files into `dest`, keeping their relative paths. Returns (written, skipped)."""
    root = bundled_root()
    if root is None:
        raise FileNotFoundError("this installation has no bundled configuration")
    written, skipped = [], []
    for sub, pattern in BUNDLED:
        for src in sorted((root / sub).glob(pattern)):
            out = Path(dest) / sub / src.name
            if out.exists() and not force:
                skipped.append(out)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, out)
            written.append(out)
    return written, skipped
