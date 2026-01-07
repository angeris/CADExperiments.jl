# TODO

## Scope & API
- Define a minimal public API with explicit state/workspace:
  - `state, work = initialize(problem, x0; options=Options())`
  - `stats = solve!(state, problem, work; options=Options())`
- Choose problem representation: residual callback `r!(out, x)` and sparse Jacobian callback `J!(J, x)` with fixed sparsity.
- Decide on outputs: final parameters in `state.x`, plus `Stats` (iters, status, residual norm, step norm).
- API sketch (proposed, adjust as needed):
  ```julia
  struct Problem
      r!::Function          # r!(out, x)::Nothing
      J!::Function          # J!(J, x)::Nothing
      m::Int
      n::Int
      jac_pattern::SparseMatrixCSC{Float64,Int}
  end

  struct Options
      max_iters::Int
      atol::Float64; rtol::Float64; gtol::Float64; step_tol::Float64
      lambda_init::Float64; lambda_min::Float64; lambda_max::Float64
  end

  struct Workspace
      r::Vector{Float64}
      J::SparseMatrixCSC{Float64,Int}
      A::SparseMatrixCSC{Float64,Int}  # J'J + λI
      g::Vector{Float64}               # J'r
      step::Vector{Float64}
      x_trial::Vector{Float64}
      diag::Vector{Float64}
  end

  struct State
      x::Vector{Float64}
      stats::Stats
  end

  struct Stats
      iters::Int
      cost::Float64
      grad_norm::Float64
      step_norm::Float64
      status::Symbol
  end
  ```

## Core Levenberg-Marquardt
- Implement the LM loop: residual/Jacobian eval, normal equations, damping update, and step acceptance.
- Use a sparse linear solver and reuse its factorization if the sparsity pattern is constant.
- Add convergence criteria: gradient norm, step norm, and relative cost decrease.

## Performance (No Allocations)
- Create a workspace struct with all buffers (residuals, Jacobian storage, normal matrix, rhs, step, temp).
- Ensure all hot paths are `!` functions and use in-place math.
- Precompute structure-dependent buffers and reuse across iterations.

## Robustness & Numerics
- Add safeguards for singular/ill-conditioned systems (damping floor/ceiling, fallback steps).
- Handle stalled progress with backtracking and max-iteration exit.
- Support scaling of parameters and residuals to improve conditioning.

## Tests & Benchmarks
- Add basic unit tests for convergence on small synthetic problems.
- Add regression tests for edge cases (rank deficiency, tiny residuals, large damping).
- Add microbenchmarks to verify zero/near-zero allocations per iteration.
- Use `test/runtests.jl` for the main harness and `bench/allocations.jl` for a fast allocation smoke check.

## Future CAD Constraints Integration
- Map common sketch constraints into residual/Jacobian callbacks.
- Keep constraint evaluation separated from solver core to allow reuse in the CAD front-end.
