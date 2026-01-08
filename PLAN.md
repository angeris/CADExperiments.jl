# Monorepo Plan (Draft)

## Goal
Restructure this repo into a monorepo with two Julia packages:
- `SparseLNNS` (solver)
- `CADConstraints` (constraint system that depends on `SparseLNNS`)

## Proposed Layout
- `SparseLNNS/` (existing solver package)
- `CADConstraints/` (new package)
- root README describes how to run tests for each package

## Dependency Strategy (decide before coding)
Pick one of these and stick to it:
1) **Monorepo dev path (simple for local):**
   - Use `Pkg.develop(path="../SparseLNNS")` in CADConstraints workflows.
2) **Git submodule for SparseLNNS (explicit external dep):**
   - Add `SparseLNNS` as a git submodule and use `Pkg.develop(path=".../SparseLNNS")`.
3) **Git URL dependency with subdir (no submodule):**
   - Use `Pkg.add(PackageSpec(url="https://github.com/angeris/CADExperiments.jl.git", subdir="SparseLNNS"))`.

## Steps (after choosing dependency strategy)
1) Move the existing SparseLNNS package into `SparseLNNS/`.
2) Create `CADConstraints/` with a minimal API and test.
3) Update root docs/AGENTS/README to describe monorepo usage.
4) Verify both packages: `Pkg.test()` in each.

## Checkpoints
- Confirm the dependency strategy.
- Confirm expected public API for CADConstraints (what types/functions should exist).
- Decide whether to keep solver tests/benchmarks in the monorepo or move them elsewhere.
