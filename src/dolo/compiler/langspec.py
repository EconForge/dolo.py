"""
``langspec`` — TEMPORARY compatibility shim around ``dolang``.

While dolo's model importer is being migrated to ``dyno.dynspec``, the only
dolang utility the new code path still uses is:

- ``eval_data``: recursively evaluates YAML section content (calibration
  values, domains, grid options, tagged process objects) against a symbol
  dictionary.

The tagged-object registration itself (``language_element``) happens in the
legacy modules ``dolo.compiler.objects``, ``dolo.numeric.processes`` and
``dolo.numeric.distribution``, which still import dolang directly.

RULES
-----
* New dolo code must obtain dolang functionality through this module
  (``from dolo.compiler import langspec``), never by importing dolang
  directly.
* Shrink, don't grow: once ``eval_data`` is reimplemented in-house (it
  only resolves a handful of tagged YAML objects) this module is deleted
  entirely.
"""

from dolang.language import eval_data

__all__ = ["eval_data"]
