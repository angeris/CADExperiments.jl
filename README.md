# Monorepo Overview

This repository hosts two Julia packages:

- `SparseLNNS`: sparse nonlinear least squares solver (Levenberg–Marquardt).
- `CADConstraints`: CAD‑style constraint system built on top of `SparseLNNS`.

## Quick commands

```sh
# SparseLNNS tests
julia --project=SparseLNNS -e 'using Pkg; Pkg.test()'

# CADConstraints tests (pull SparseLNNS from the repo subdir)
julia --project=CADConstraints -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/angeris/SparseLNNS.jl.git", subdir="SparseLNNS")); Pkg.test()'
```
