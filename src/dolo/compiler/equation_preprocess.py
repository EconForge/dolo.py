"""
TEMPORARY module : normalization of dolo-style equation strings.

The end game (see the migration plan in the repo history / PR) is that
*all* equation parsing happens in ``dyno.dynspec``.  Complementarity
conditions (``⟂`` / legacy ``|``) are already parsed natively by
dynospec (``albop/dynspec`` branch) and split structurally by
``model_v2._complementarity_parts``.

What remains in this module is dolo-only *timing* syntax that dynospec
deliberately does not support (untimed symbols are constants there):
parenthesis timing ``x(1)`` and untimed-symbol dating ``x`` → ``x[t]``,
plus ``==`` normalization.  Rewritten line by line into plain
dynspec-grammar text before parsing.

Everything in here is considered throwaway; do NOT import from this module
outside of ``dolo.compiler.model_v2``.
"""

from __future__ import annotations

import re

_timing_renamer = re.compile(
    r"(?<![\w])(?:(?P<neg>\w+)\(\s*-\s*(?P<nd>\d+)\s*\)|(?P<name>\w+)\(\s*(?P<d>-?\d+)\s*\))"
)


def rewrite_timing_calls(line: str, symbols: set[str]) -> str:
    """Rewrite dolo parenthesis timing ``x(±d)`` into dynspec ``x[t±d]``.

    Only known symbol names are touched, so function calls (``log(2)``)
    and other parentheses pass through untouched.  Unlike
    :func:`normalize_timing`, untimed symbols are *not* suffixed with
    ``[t]`` — used where plain names must remain constants
    (``eval_formula``).
    """

    def _ren(m):
        name = m.group("neg") if m.group("neg") is not None else m.group("name")
        if name not in symbols:
            return m.group(0)
        d = int(m.group("nd") if m.group("nd") is not None else m.group("d"))
        if d == 0:
            return f"{name}[t]"
        return f"{name}[t+{d}]" if d > 0 else f"{name}[t{d}]"

    return _timing_renamer.sub(_ren, line)


def normalize_timing(line: str, symbols: set[str]) -> str:
    """Rewrite dolo timing conveniences into dynspec `x[t±d]` syntax.

    - ``x(1)`` / ``x(-1)`` (legacy parenthesis timing) where ``x`` is a
      known symbol becomes ``x[t+1]`` / ``x[t-1]``;
    - an untimed known symbol means "date 0": ``e_z`` becomes ``e_z[t]``.
    """

    def _named(m):
        name = m.group("name")
        d = int(m.group("d"))
        return f"{name}[t+{d}]" if d != 0 else f"{name}[t]"

    def _named_neg(m):
        name = m.group("neg")
        return f"{name}[t-{m.group('nd')}]"

    def _ren(m):
        name = m.group(0)
        core = re.match(r"(\w+)", name).group(1)
        if core in symbols:
            return _named_neg(m) if m.group("neg") else _named(m)
        return name

    line = _timing_renamer.sub(_ren, line)

    # Untimed symbols -> [t].  Only known symbols are touched, so numbers and
    # function names (log, exp, ...) are left alone.
    def _untimed(m):
        name = m.group("name")
        if name in symbols:
            return f"{name}[t]"
        return name

    line = re.sub(
        r"(?<![\w])(?P<name>[A-Za-zΑ-Ωα-ω_][\w]*(?:′|\')*)(?!\s*\[)", _untimed, line
    )

    # dolo accepts `==` and `=` interchangeably for assignments.
    line = line.replace("==", "=")
    return line


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
