# dynospec requirements — dolo `⟂` complementarity support

Handover note for the dyno.py side of the dolang → dynspec transition.
**dolo.py side:** branch `feat/dyno-importer` of `dolo.py`, status in
`DEV_NOTES.md` (this repo). **dyno.py side:** today's `dyno.dynspec`
(grammar `grammars/grammar.lark`, analyzer `analyze.py`, recipes
`recipe.py`, file parser `larkfiles.py`).

The target artifact this enables:
[dolo.py: examples/dyno/rbc_dolo_comp.dyno] — a `.dyno` model using the
syntax described below. Once implemented, that file must load through
`ModelV2.from_dyno` and reproduce `examples/models/rbc.yaml`.

## Status (2026-10-16)

Implemented on dyno.py `albop/dynspec`:

- `1874e94` — §2.1 grammar (`complementarity` rule, `⟂`/`|`, TimeFixer
  normalization to `complementarity(residual, lb, x, ub)` with explicit
  `-inf`/`inf`), residual-only evaluation /
  `extract_variables_from_equation`, `\perp` latex, built-in `inf`,
  example file + tests.
- `e3596bc` — **§4 recipes**: YAML recipe loading, `check_group` /
  `check_model`, complementarity conformity (i-th complementarity
  bounds the i-th control; bounds restricted to the spec), bundled
  `recipes.yaml` with `arbitrage_bounds`, `controls_lb`/`controls_ub`
  groups, and `extract_complementarity`/`extract_rhs_variables`.

**dolo consumes all of it**: group specs come from `DTCC_RECIPE` (its
`_BOUND_SPEC` is deleted), complementarity conformity produces the
compile-time errors, `extract_complementarity`/`extract_rhs_variables`
drive `.dyno` symbol inference, and both are committed tests
(`test_dyno_comp_import_rbc`, `test_dyno_standalone_bounds_sections`,
`test_complementarity_positional_check`).

Still open:
- §2.3 — optional `x(1)` timing canonicalization utility + `==` alias
  (untimed→`[t]` is a confirmed non-goal).

---

## 1. Why

dolo's new importer (`dolo.compiler.model_v2`) parses model text with
dynspec, but dolo models carry three syntax features dynspec does not
have yet. dolo currently rewrites that text *before* parsing in a
throwaway module, `dolo/compiler/equation_preprocess.py` (TEMP). The
goal of this work: make the grammar natively support the syntax so
`equation_preprocess.py` can be deleted. (No dolang code is involved on
the new path; dolang is out of scope here.)

## 2. Syntax to support

### 2.1 `⟂` complementarities on equations

An equation in a block may carry a complementarity suffix:

```
<residual>  ⟂  <lb> <= <x>[t] [<= <ub>]
```

- `<residual>` is the equation as today (`bare_formula` or `equality`).
- The bound is attached to exactly one control variable `<x>` at shift 0.
- `<lb>` / `<ub>` are formulas. Omitted lower bound → `-inf`, omitted
  upper bound → `+inf`, so these are all valid and equivalent to the
  full form where the blank side is `±inf`:

```
g  ⟂ 0.0 <= n[t] <= inf
g  ⟂ 0.0 <= n[t]                         # one-sided: ub = +inf
g  ⟂ -inf <= n[t] <= 1.0                 # lb = -inf
```

- **Positional meaning (dolo convention):** in the `[arbitrage]` block
  the i-th equation's complementarity must involve the i-th control
  (in the model's `controls` declaration order). Violating this is a
  conformity error with a clear message ("expected `<control_i>[t]`,
  found `<other>`").
- Legacy separator `|` (pre-`⟂` dolo syntax) — *optional*: dolo only
  treats `|` as a separator when it is followed by a comparison
  (`... | 0 <= x[t] <= 1`), otherwise it is the (deprecated) OR
  operator. Supporting it keeps old YAML files on the new parser;
  `⟂` is the recommendation going forward.

Grammar anchors (today, `dyno/dynspec/grammars/grammar.lark`):

- `?equation: equality | bare_formula` (≈line 31) — add a third
  alternative,
- terminal `COMPARISON: ("<="|"<"|">"|">=")` already exists (≈line 40)
  but is only used by `t_double_bound` (line 41), a useful precedent for
  the chained bound form.

Suggested shape (exact AST at your discretion):

```
?equation: equality | bare_formula | complementarity
complementarity: formula "⟂" comparisons
comparisons:     formula COMPARISON variable -> bound (one-sided)
               | formula COMPARISON variable COMPARISON formula -> bounds
```

with a kept tree so `extract_*`/`str_expression`/`latex` can traverse
it (see §3).

### 2.2 Standalone bounds sections

dolo also allows bounds in dedicated equation groups (dolo docs
convention), keyed by block tag:

```
[controls_lb] :: {
    n[t] = 0.0
    i[t] = -inf
}
[controls_ub] :: {
    n[t] = 1.0
    i[t] = (1-delta)*k[t] + k[t]^alpha
}
```

- Each line is an `equality` whose LHS is a *control* variable at
  shift 0 (already legal grammar — `equality: formula "=" formula`);
  the RHS is the bound expression.
- Bounds (here and in §2.1) may only involve `exogenous[t]`,
  `states[t]`, and `parameters`. **No controls, no auxiliaries, no
  time shifts.** This is the bounds conformity spec (§4).
- There are exactly `len(controls)` equations.
- `rbc_dolo_comp.dyno` shows both syntaxes; a model uses either the
  `⟂` suffix form or the standalone sections (dolo accepts both).

### 2.3 dolo timing conveniences — *optional tier*

These need symbol knowledge, so they are post-parse utilities rather
than grammar (grammar can't know `exp` is a function while `k` is a
variable):

- parenthesis timing `x(1)` → `x[t+1]`, `x(-1)` → `x[t-1]`;
- `exp(2)` / `log(x)` must **not** be rewritten (function calls with
  numeric/expr arguments are calls).

**Non-goal: untimed variables.** A variable name without a time index is
a constant/parameter in dynspec, and that stays. The "untimed symbol
means date 0" convention exists only in old dolo-YAML files; dolo keeps
normalizing *those* in its own YAML-path shim
(`dolo/compiler/equation_preprocess.py`) until that legacy syntax is
gone. Do not add implicit `[t]` to dynspec.

Suggested placement: an `analyze`/`symbolic` utility
`canonicalize_timing(tree_or_text, variable_names)` + re-parse, so dolo
can drop its regex `normalize_timing` / `rewrite_timing_calls`.

Also trivial: accept `==` wherever `=` is accepted today (dolo files use
both for equality/assignment). One-line grammar alias.

## 3. Required API surface (what dolo will import)

Minimum so that `dolo.compiler.model_v2` can drop the text-level
preprocessing:

1. **Parse.** `parser.parse(txt, start="equation_block" |
   "complementarity_block")` over the block contents handles every form
   in §2.1.
2. **Read the structure.** An analyzer helper à la
   `extract_lhs_variable`:
   `extract_complementarity(tree) -> (lb_tree | None, variable_name,
   ub_tree | None)` (or a small dataclass with `.lower`, `.upper`,
   `.variable`, `.shift`), returning `None` for plain equations. Both
   bound expressions as trees so they can be parsed/stringified/conformity
   checked.
3. **Traversal.** `extract_variables_from_equation` must walk the
   complementarity subtree (bound expressions count as occurrences).
   `str_expression` renders the residual *only* (dolo's `equations`
   dict shows `residual` and keeps `arbitrage_lb`/`arbitrage_ub` as
   separate groups). Optionally expose full-fidelity rendering.
4. **Latex.** `latex` renders `⟂` as `\perp` between residual and
   bounds (or at minimum renders the residual — dolo's `_repr_html_`
   shows bounds-less rows).
5. **Conformity.** `check_equation_group(..., spec)` on an `arbitrage`
   group containing complementarities must verify §2.1's positional
   rule, one bound per equation, and run the bounds-conformity check of
   §4 against the `arbitrage_bounds` spec; on failure the `CheckResult`
   lists violations like today. The recipes change of §4 is what makes
   `arbitrage_lb`/`arbitrage_ub` first-class.
6. **Tags pass through.** `DynoFile` (larkfiles.py) already preserves
   block tags in `statement_metadata`; verify `[controls_lb]` /
   `[controls_ub]` blocks survive as tagged equations (dolo classifies
   equations by those tags today — `ModelV2.from_dyno`).

## 4. Recipe change

`dolo/compiler/functions_v2.py` currently hardcodes what should become a
dynspec group spec (module-level `_BOUND_SPEC`):

```python
EquationGroupSpec(
    name="arbitrage_bounds",
    allowed=[
        VariableSpec("exogenous", 0),
        VariableSpec("states", 0),
        VariableSpec("parameters", 0),
    ],
    target=None, recursive=False,
    n_equations="controls", optional=True,
)
```

Land the equivalent in `dyno.dynspec.recipe` (add to `DTCC_RECIPE`),
covering both `arbitrage`-attached complementarities and the standalone
`controls_lb`/`controls_ub` sections. dolo then:
- `check_equation_group` with this spec for bounds conformity,
- `compile_group` bounds exactly like any other group.

## 5. What dolo will do once this lands (its cost, not yours)

- delete `dolo/compiler/equation_preprocess.py`;
- `ModelV2.from_yaml` / `from_dyno` switch from `split_complementarity`
  + `normalize_timing` text surgery to the parsed complementarity
  structure (`extract_complementarity`) and dynspec recipes;
- `functions_v2._BOUND_SPEC` becomes
  `DTCC_RECIPE.get_group_spec("arbitrage_bounds")`.

## 6. Acceptance criteria

1. `dolo.py` loads `examples/dyno/rbc_dolo_comp.dyno` with
   `ModelV2.from_dyno` with **no preprocessing**, giving
   `symbols`, `calibration`, `functions["arbitrage"]` +
   `["arbitrage_lb"]`/`["arbitrage_ub"]` identical to
   `yaml_import_v2("examples/models/rbc.yaml")`.
2. Bounds function values equal the YAML-twin bounds (`0.0` / `inf`
   rows for this RBC).
3. Error quality: wrong control in a complementarity raises a
   conformity error naming expected vs found variable; a bound using
   `controls`/`auxiliaries` or a shifted variable reports a violation.
4. Existing dyno tests still pass (plain equations keep working;
   `|`-free files are unaffected).
5. dolo's migration contract still green: `pixi run --environment dev
   pytest -q` (54 tests at the time of writing), then
   `equation_preprocess.py` is deleted as a follow-up.

## 7. Decisions to take on your side / non-goals

- Exact AST node naming/shape for the complementarity (kept vs. filtered
  rules) — your call; dolo consumes it through the helpers of §3.
- Legacy `|` and `x(1)` support are *optional*; skip them and old-style
  YAML files just keep a thin dolo-side shim until the next release.
  Untimed variables are a **non-goal** (§2.3): dynospec never adds
  implicit `[t]`; that normalization stays on dolo's YAML legacy path.
- dolang, the legacy `Model` importer, and YAML-tag objects are out of
  scope here.

Reference: dolo consumers to keep in mind — `model_v2._parse_group_trees`
(YAML comp splitting), `model_v2.from_dyno` (tag classification),
`functions_v2._BOUND_SPEC`, `_emit`'s `equality` handling, and the tests
`tests/test_model_v2.py::test_dyno_import_rbc`,
`::test_standalone_bounds_groups`, `::test_parity_with_legacy_importer`.
