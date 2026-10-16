"""
TEMPORARY module : normalization of dolo-style equation strings.

The end game (see the migration plan in the repo history / PR) is that
*all* equation parsing happens in ``dyno.dynspec``, with the ``⟂``
complementarity operator and dolo's timing conventions added to the
dynspec grammar itself.  Until then, dolo YAML models keep their current
syntax (``⟂`` bounds, legacy ``|`` bounds, ``==``, parenthesis timing
``x(1)`` and untimed symbols such as ``e_z``) and this module rewrites
them, line by line, into plain dynspec-grammar text.

Everything in here is considered throwaway; do NOT import from this module
outside of ``dolo.compiler.model_v2``.
"""

from __future__ import annotations

import re


def split_complementarity(
    line: str, control: str | None = None
) -> tuple[str, str | None, str | None]:
    """Split a dolo equation carrying a complementarity bound.

    An equation can be of the forms:

        <residual> ⟂ <lb> <= x[t] <= <ub>     (recommended)
        <residual> ⟂ <lb> <= x[t]             (legacy, one-sided)
        <residual> | <lb> <= x[t] <= <ub>     (legacy separator)

    where the bound expressions contain the current *control* variable.
    Returns ``(residual, lower_bound, upper_bound)``; bounds are ``None``
    when there is no complementarity on the line.
    """
    s = line

    # Locate the separator: "⟂" always wins over "|" (the latter is only
    # treated as a separator when it is followed by an inequality).
    sep = None
    if "⟂" in s:
        sep = "⟂"
    elif "|" in s:
        head, tail = s.split("|", 1)
        if "<=" in tail or ">=" in tail:
            sep = "|"
        else:
            return s, None, None
    else:
        return s, None, None

    residual, comp = [e.strip() for e in s.split(sep, 1)]
    comp = comp.strip()

    # Expected middle term: the control variable, optionally timed.
    pattern = re.compile(
        r"^(?P<lb>.*?)\s*<=\s*(?P<var>\w+)\s*(\[(?:t[+-]?\d*)\]\s*)?"
        r"(?:<=\s*(?P<ub>.*))?$"
    )
    match = pattern.match(comp)
    if match is None:
        raise Exception(f"Unable to parse complementarity condition '{comp}'")
    lb, middle, ub = match.group("lb").strip(), match.group("var"), match.group("ub")
    if control is not None and middle != control:
        raise Exception(
            f"Complementarity condition '{comp}' incorrect. "
            f"Expected '{control}' instead of '{middle}'."
        )
    if lb == "":
        lb = "-inf"
    ub = (ub or "inf").strip()
    if ub == "":
        ub = "inf"
    return residual, lb, ub


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
