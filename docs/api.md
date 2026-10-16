# API Reference

## Top-level API

The top-level namespace of `dolo` re-exports the model import and
solution routines.

::: dolo

## Model

Model objects, produced by `yaml_import`, are the central piece of the
package: they hold the parsed specification, the calibration, and the
compiled functions of the model.

::: dolo.compiler.model

## Importing models

::: dolo.compiler.model_import

## Steady state and solution algorithms

::: dolo.algos.time_iteration
