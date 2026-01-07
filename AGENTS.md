# Repository Guidelines

## Project Structure & Module Organization
- `Project.toml` defines the Julia package and dependencies.
- `src/NLLS.jl` holds the main module. Keep public APIs small and export only stable entry points.
- Tests are not present yet; plan to add a `test/` directory with `runtests.jl` once solver logic lands.

## Build, Test, and Development Commands
- `julia --project -e 'using Pkg; Pkg.instantiate()'` installs dependencies for this repo.
- `julia --project -e 'using Pkg; Pkg.test()'` runs the test suite (once `test/` exists).
- `julia --project` starts a REPL with this environment for local development.

## Coding Style & Naming Conventions
- Use 4-space indentation, no tabs. Keep lines short and readable.
- Prefer small, composable functions with clear in-place variants (e.g., `solve!(state, ...)`).
- Follow Julia naming: modules and types in `CamelCase`, functions in `lowercase_with_underscores` only when needed.
- Keep allocations out of hot paths; preallocate buffers and reuse them across iterations.

## Testing Guidelines
- Use Julia’s `Test` standard library (`using Test`).
- Name tests by behavior, e.g., `@testset "LM damping updates"` once implemented.
- Add regression tests for numerical edge cases (singular Jacobian, near-zero residuals, stalled steps).

## Commit & Pull Request Guidelines
- There is no commit history yet, so no established convention to mirror.
- Use short, imperative commit summaries (e.g., “Add LM step acceptance rule”).
- PRs should include: a brief problem statement, approach, and any new benchmarks or tests.

## Performance & Allocation Discipline
- Avoid heap allocations in iterations: reuse workspace arrays, avoid temporary sparse matrices.
- Prefer explicit workspace structs and `!` methods to make data flow and mutability obvious.
