# Monorepo Overview

This repository hosts three Julia packages and is an experimental playground for a future CAD solver stack (we plan a Rust rewrite later):

- `SparseLNNS`: sparse nonlinear least squares solver (Levenberg–Marquardt).
- `CADConstraints`: CAD‑style constraint system built on top of `SparseLNNS`.
- `CADSketchUI`: CImGui-based sketch UI that exercises `CADConstraints`.

## Quick commands

```sh
# SparseLNNS tests
julia --project=SparseLNNS -e 'using Pkg; Pkg.test()'

# CADConstraints tests (pull SparseLNNS from the repo subdir)
julia --project=CADConstraints -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/angeris/SparseLNNS.jl.git", subdir="SparseLNNS")); Pkg.test()'

# CADSketchUI demo (pull CADConstraints + SparseLNNS from repo subdirs)
julia --project=CADSketchUI -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/angeris/SparseLNNS.jl.git", subdir="SparseLNNS")); Pkg.add(PackageSpec(url="https://github.com/angeris/SparseLNNS.jl.git", subdir="CADConstraints")); using CADSketchUI; CADSketchUI.run()'
```
