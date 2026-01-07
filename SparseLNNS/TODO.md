# TODO

## Scope & API
- Define a minimal public API with explicit state/workspace:
  - `state, work = initialize(problem, x0; options=Options())`
  - `stats = solve!(state, problem, work; options=Options())`
- Choose problem representation: residual callback `r!(out, x)` and sparse Jacobian callback `J!(J, x)` with fixed sparsity.
- Decide on outputs: final parameters in `state.x`, plus `Stats` (iters, status, residual norm, step norm).
- API sketch (proposed, adjust as needed):
  ```julia
  struct Problem{T}
      r!::Function          # r!(out, x)::Nothing
      J!::Function          # J!(J, x)::Nothing
      m::Int
      n::Int
      jac_pattern::SparseMatrixCSC{T,Int}
  end

  struct Options{T}
      max_iters::Int
      atol::T; rtol::T; gtol::T; step_tol::T
      lambda_init::T; lambda_min::T; lambda_max::T
  end

  struct Workspace{T}
      r::Vector{T}
      J::SparseMatrixCSC{T,Int}
      A::SparseMatrixCSC{T,Int}  # J'J + λI
      g::Vector{T}               # J'r
      step::Vector{T}
      x_trial::Vector{T}
      diag::Vector{T}
  end

  struct State{T}
      x::Vector{T}
      stats::Stats{T}
  end

  struct Stats{T}
      iters::Int
      cost::T
      grad_norm::T
      step_norm::T
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
