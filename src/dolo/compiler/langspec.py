"""
``langspec`` — TEMPORARY compatibility shim around ``dolang``.

While dolo's model importer is being migrated to ``dyno.dynspec``, we keep
using a handful of dolang utilities for the parts that dynspec does not
cover yet:

- ``language_element``: registers tagged YAML objects (``!Normal``,
  ``!Product``, ``!Conditional`` ...) as PyYAML constructors;
- ``eval_data``: recursively evaluates YAML section content (calibration
  values, domains, grid options) against a symbol dictionary;
- ``solve_triangular_system``: solves symbolic equations in dependency order;
- ``greek_tolerance``: decorator accepting greek-letter keyword arguments;
- ``ModelError``: the error type raised for model specification mistakes.

RULES
-----
* New dolo code must obtain this functionality through this module
  (``from dolo.compiler import langspec``), never by importing dolang
  directly.
* Each entry point should eventually be reimplemented in-house (or moved to
  ``dyno.dynspec``) and removed from this module.  Keep the module small:
  it is a parking lot, not a home.
"""

# Re-export the dolang language machinery (lazy imports keep dolo's top-level
# import cheap; dolang itself becomes optional once these are replaced).
from dolang.language import language_element, eval_data, ModelError, greek_tolerance
from dolang.triangular_solver import solve_triangular_system

__all__ = [
    "language_element",
    "eval_data",
    "ModelError",
    "greek_tolerance",
    "solve_triangular_system",
]
