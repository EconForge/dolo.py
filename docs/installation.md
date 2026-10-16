## Basic installation

Dolo can be installed in several ways:

- with pixi, using the `econforge` channel (recommended):

  ```console
  pixi workspace channel add --prepend https://prefix.dev/econforge
  pixi add dolo
  ```

- with anaconda:

  `conda install -c conda-forge dolo`

- with pip

  `pip install dolo`


## Developper's installation

Contributors work from a clone of the repository. Dolo uses `pixi` to manage
the development environment:

```console
pixi install -e dev    # test, lint, docs, and release features
pixi run -e dev test  # unit tests
pixi run -e dev cov     # unit tests with coverage
pixi run -e dev docs  # serve the documentation locally
```
