# Monorepo Overview

This repository hosts two Julia packages:

- `packages/SparseLNNS`: sparse nonlinear least squares solver (Levenberg–Marquardt).
- `packages/CADConstraints`: CAD‑style constraint system built on top of `SparseLNNS`.

## Quick commands

```sh
# SparseLNNS tests
julia --project=packages/SparseLNNS -e 'using Pkg; Pkg.test()'

# CADConstraints tests (develop SparseLNNS from the monorepo)
julia --project=packages/CADConstraints -e 'using Pkg; Pkg.develop(path="../SparseLNNS"); Pkg.test()'
```
