Informations for developpers
================================

Current folder organization

- model file (dynare modfile or yamlfile)
- symbolic model (sympy only)
- compiler (sympy only)
- numeric (everything with numerical calculations)

New structure

1. serialized model: YAML file / modfile

2. symbolic model
    - generic
    - implements type checkers

3. compiled models (complies with numeric interface)
    - can be Matlab/Python/...

4. numerical methods
    - pure algorithms: take matrices as input
    - model aware: take compiled models as input

5. commands
    - define user-friendly commands