# SparseLNNS

SparseLNNS is a tiny, allocation‑aware Levenberg–Marquardt solver for sparse nonlinear least squares in Julia. It is designed for fixed sparsity patterns and fast repeated solves.

## Quick start

```julia
using SparseArrays
using SparseLNNS

r!(out, x) = (out[1] = x[1] - 1.0)
function J!(J, x)
    nonzeros(J)[1] = 1.0
    return nothing
end

Jpat = spzeros(1, 1)
Jpat[1, 1] = 1.0
prob = Problem(r!, J!, Jpat)

state, work = initialize(prob, [0.0])
stats = solve!(state, prob, work)

println("x = ", state.x[1], " status = ", stats.status)
```

## Notes
- The Jacobian sparsity pattern is fixed; `J!` must fill values in place for that pattern.
- Use `initialize` once and reuse `state`/`work` for repeated solves.
- See `docs/EXAMPLES.md` for more examples and CAD‑inspired constraints.
