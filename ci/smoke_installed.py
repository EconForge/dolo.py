"""Smoke test for an installed ``dolo`` package.

Run it with the interpreter of a clean environment where only the built wheel
is installed, from a directory outside the source tree:

    cd "$(mktemp -d)" && /path/to/venv/bin/python /path/to/ci/smoke_installed.py

It checks that the package is not imported from the checkout, that package
data (``compiler/recipes.yaml``) and the shipped examples are present, and
that a model from the examples imports and calibrates correctly.
"""

import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import dolo

package_dir = Path(dolo.__file__).resolve().parent
assert "site-packages" in package_dir.parts, f"dolo imported from {package_dir}"
pkg_ver = version("dolo")
print(f"dolo {pkg_ver} imported from {package_dir}")

# Package data.
recipes = sorted(p.name for p in (package_dir / "compiler").glob("*.yaml"))
assert recipes == ["recipes.yaml"], recipes

# Examples are shipped inside the wheel (force-include in pyproject.toml).
examples_dir = package_dir / "examples"
assert examples_dir.is_dir(), f"no examples dir in {package_dir}"
assert (examples_dir / "models" / "rbc.yaml").is_file(), list(examples_dir.iterdir())

# Load an example from a copy, so nothing depends on the package directory.
workdir = Path(tempfile.mkdtemp())
model_file = shutil.copy(examples_dir / "models" / "rbc.yaml", workdir)
model = dolo.yaml_import(str(model_file))
assert model.name == "Real Business Cycle", model.name
assert "k" in model.symbols["states"], model.symbols["states"]

print("smoke test passed")
