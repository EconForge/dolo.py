"""
TEMPORARY module : normalization of dolo-style equation *sections*.

The end game (see the migration plan in the repo history / PR) is that
*all* equation parsing happens in ``dyno.dynspec``.  Complementarity
conditions (``⟂`` / legacy ``|``) and block conformity are already
handled natively by dynospec (dyno.py `albop/dynspec` branch).

What remains in this module is dolo-*YAML*-specific section handling:

- ``split_equations``: split an ``equations:`` section entry (block scalar
  or list) into single-line strings, stripping ``#`` comments;
- ``normalize_line``: accept ``==`` wherever dynospec expects ``=``
  (legacy dolo files use both).

Timing conveniences (`x(1)` parenthesis timing, untimed symbols read as
date 0) are **no longer supported**: timing must be explicit (`x[t]`,
`x[t-1]`, `x[t+1]`), and a name without ``[t]`` is a constant.

Everything in here is considered throwaway; do NOT import from this module
outside of ``dolo.compiler.model_v2``.
"""

from __future__ import annotations


def normalize_line(line: str) -> str:
    """Accept legacy ``==`` in place of dynospec's ``=``."""
    return line.replace("==", "=")


def split_equations(block) -> list[str]:
    """Turn an `equations:` section entry into a list of single-line strings."""
    if isinstance(block, str):
        lines = [ln.strip() for ln in block.splitlines()]
    else:
        lines = list(block)
    out = []
    for ln in lines:
        ln = str(ln).strip()
        # strip comments (dolo models use `#`-comments aligned to the right)
        if "#" in ln:
            ln = ln.split("#", 1)[0].rstrip()
        if ln:
            out.append(ln)
    return out
