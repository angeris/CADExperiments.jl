# Repository Guidelines

## Project Structure & Module Organization
- `Project.toml` defines the Julia package and stdlib deps.
- `src/SparseLNNS.jl` contains the LM solver and types. Keep public exports minimal.
- `test/runtests.jl` hosts unit tests (basic solves, Rosenbrock, CAD-inspired constraints, allocations).
- `bench/` contains quick scripts for allocations and timing sanity checks.

## Build, Test, and Development Commands
- `julia --project -e 'using Pkg; Pkg.instantiate()'` sets up the environment.
- `julia --project -e 'using Pkg; Pkg.test()'` runs the full test suite.
- `julia --project bench/allocations.jl` reports per-call allocations.
- `julia --project bench/linear_timing.jl` runs a 1000x1000 sparse linear timing check.

## Coding Style & Naming Conventions
- Use 4-space indentation, no tabs. Keep functions short and composable.
- Prefer in-place APIs (`solve!`, `r!`, `J!`) with preallocated buffers.
- Types in `CamelCase`, functions in `lowercase`. Avoid clever metaprogramming.
- Add brief comments when logic is non-obvious or numerically delicate.

## Performance & Allocation Discipline
- Avoid allocations in the LM loop: preallocate residuals, Jacobian storage, steps, and RHS.
- Keep the augmented system sparse and fixed-pattern; only update `nzval` data.
- Favor deterministic data flow over implicit global state.

## Testing Guidelines
- Use Julia `Test` and keep tests deterministic with fixed seeds when random.
- Include at least one known-solution test for any new feature.
- Add a regression test for any numerical stability fix.

## Commit & PR Guidelines
- Use short, imperative commit messages.
- PRs should describe the problem, the approach, and include tests/bench output when relevant.
