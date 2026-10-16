"""Check that the runtime dependencies declared in pixi.toml match pyproject.toml.

``pyproject.toml`` is the single source of truth for the package version, the
Python floor (``requires-python``) and the runtime dependencies: hatchling
builds the wheel from it and pixi-build-python reads it to build the conda
package. ``pixi.toml`` still lists the runtime dependencies twice, for the
development workspace (``[dependencies]``) and for the ``min-deps``
environment that pins every dependency to its floor. This script fails when
those copies drift apart:

- every ``[project.dependencies]`` entry appears in ``[dependencies]`` with the
  same version range;
- ``python`` in ``[dependencies]`` matches ``requires-python``;
- ``[feature.min-deps.dependencies]`` pins each dependency to the lower bound
  from ``pyproject.toml`` and Python to its lowest supported minor version;
- each optional extra matches the pixi feature of the same name;
- ``[package]`` does not redeclare the version or run dependencies.

Run it with ``python ci/check_deps.py`` (Python >= 3.11, standard library only).
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# PyPI name -> conda-forge name, where they differ.
CONDA_NAMES: dict[str, str] = {}

# PyPI distribution name -> canonical project name, where they differ.
# `dynopy` is the PyPI publishing name of the `dyno` project: pyproject.toml
# declares the PyPI name, while the dev workspace installs the local checkout
# under its canonical project name (`dyno = { path = ... }`).
PYPI_ALIASES: dict[str, str] = {"dynopy": "dyno"}

# Packages where the pyproject.toml floor cannot be installed in the
# min-deps environment (e.g. pyyaml 6.0 has no conda-forge build for
# Python >=3.12).  Values are the pin that min-deps should use instead.
MIN_DEPS_OVERRIDES: dict[str, str] = {
    # pyyaml 6.0 predates Python 3.12 builds on conda-forge.
    "pyyaml": "==6.0.1",
    # The first releases built for / compatible with the NumPy 2 floor.
    "matplotlib": "==3.8.4",
    "scipy": "==1.13.1",
    "xarray": "==2024.7.0",
}

REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*?)\s*$")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_spec(spec: str) -> frozenset[str]:
    """Order-insensitive set of the comma-separated clauses of a version range."""
    return frozenset(c.replace(" ", "") for c in spec.split(",") if c.strip())


def lower_bound(spec: str) -> str | None:
    for clause in normalize_spec(spec):
        if clause.startswith(">="):
            return clause[2:]
    return None


def main() -> int:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text())
    errors: list[str] = []

    project = pyproject["project"]
    requirements: dict[str, str] = {}
    for requirement in project["dependencies"]:
        match = REQUIREMENT.match(requirement)
        assert match, requirement
        name, spec = match.groups()
        conda_name = CONDA_NAMES.get(name, name)
        requirements[normalize_name(conda_name)] = spec

    workspace = {normalize_name(k): v for k, v in pixi["dependencies"].items()}
    # PyPI dependencies live in their own table.  A package may be declared
    # there instead of in ``[dependencies]`` (e.g. ``dynopy``, which only
    # exists on PyPI).  Table-valued entries (``{ path = ..., editable =
    # true }``) are development shortcuts: their presence is checked, but
    # they do not carry a version spec to compare against pyproject.toml.
    pypi_workspace = {
        normalize_name(k): v for k, v in pixi.get("pypi-dependencies", {}).items()
    }
    min_deps = {
        normalize_name(k): v
        for k, v in pixi["feature"]["min-deps"]["dependencies"].items()
    }
    min_deps_pypi = {
        normalize_name(k): v
        for k, v in pixi.get("feature", {})
        .get("min-deps", {})
        .get("pypi-dependencies", {})
        .items()
    }

    # Python floor.
    requires_python = project["requires-python"]
    if normalize_spec(workspace.get("python", "")) != normalize_spec(requires_python):
        errors.append(
            f"pixi.toml [dependencies] python = {workspace.get('python')!r} "
            f"does not match pyproject.toml requires-python = {requires_python!r}"
        )
    python_floor = lower_bound(requires_python)
    expected_python = ".".join((python_floor or "").split(".")[:2]) + ".*"
    if min_deps.get("python") != expected_python:
        errors.append(
            f"pixi.toml [feature.min-deps.dependencies] python = "
            f"{min_deps.get('python')!r}, expected {expected_python!r}"
        )

    # Runtime dependencies.
    for name, spec in requirements.items():
        if name in workspace:
            if normalize_spec(workspace[name]) != normalize_spec(spec):
                errors.append(
                    f"{name}: pixi.toml [dependencies] has {workspace[name]!r}, "
                    f"pyproject.toml has {spec!r}"
                )
        elif name in pypi_workspace:
            pypi_spec = pypi_workspace[name]
            if isinstance(pypi_spec, str):
                if normalize_spec(pypi_spec) != normalize_spec(spec):
                    errors.append(
                        f"{name}: pixi.toml [pypi-dependencies] has "
                        f"{pypi_spec!r}, pyproject.toml has {spec!r}"
                    )
            # Table entries (path deps) have no comparable version spec.
        elif PYPI_ALIASES.get(name) in pypi_workspace:
            # e.g. `dynopy` (PyPI) provided by the `dyno` path entry above.
            pypi_spec = pypi_workspace[PYPI_ALIASES[name]]
            if isinstance(pypi_spec, str) and normalize_spec(
                pypi_spec
            ) != normalize_spec(spec):
                errors.append(
                    f"{name}: pixi.toml [pypi-dependencies] has "
                    f"{pypi_spec!r}, pyproject.toml has {spec!r}"
                )
        else:
            errors.append(
                f"{name}: missing from pixi.toml [dependencies]"
                " and [pypi-dependencies]"
            )
        floor = lower_bound(spec)
        if floor is None:
            errors.append(f"{name}: no lower bound in pyproject.toml ({spec!r})")
        else:
            expected_pin = MIN_DEPS_OVERRIDES.get(name, f"=={floor}")
            pin = min_deps.get(name)
            if pin is None:
                pin = min_deps_pypi.get(name)
            if pin != expected_pin:
                errors.append(
                    f"{name}: pixi.toml [feature.min-deps] pins it to "
                    f"{pin!r}, expected {expected_pin!r}"
                )
    for name in sorted(
        (set(min_deps) | set(min_deps_pypi)) - set(requirements) - {"python"}
    ):
        errors.append(
            f"{name}: pinned in [feature.min-deps] "
            "but not a dependency in pyproject.toml"
        )

    # Optional extras mirror the pixi feature of the same name.
    features = pixi.get("feature", {})
    for extra, extra_requirements in project.get("optional-dependencies", {}).items():
        feature_deps = {
            normalize_name(k): v
            for k, v in features.get(extra, {}).get("dependencies", {}).items()
        }
        for requirement in extra_requirements:
            match = REQUIREMENT.match(requirement)
            assert match, requirement
            name, spec = match.groups()
            name = normalize_name(CONDA_NAMES.get(name, name))
            if normalize_spec(feature_deps.get(name, "")) != normalize_spec(spec):
                errors.append(
                    f"{name}: pixi.toml [feature.{extra}.dependencies] has "
                    f"{feature_deps.get(name)!r}, pyproject.toml extra "
                    f"{extra!r} has {spec!r}"
                )

    # The conda package must take its metadata from pyproject.toml.
    package = pixi.get("package", {})
    for key in ("version", "run-dependencies"):
        if key in package:
            errors.append(
                f"pixi.toml [package] declares {key!r}; pixi-build-python "
                "reads it from pyproject.toml, declare it there only"
            )

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not errors:
        print(f"OK: {len(requirements)} runtime dependencies in sync.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
