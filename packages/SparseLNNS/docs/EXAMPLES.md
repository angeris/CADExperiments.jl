# Examples

This page shows small, self‑contained setups for `SparseLNNS`.

## Scalar nonlinear solve

Solve `x^3 - 1 = 0` with a single residual.

```julia
using SparseArrays
using SparseLNNS

r!(out, x) = (out[1] = x[1]^3 - 1.0)
function J!(J, x)
    nonzeros(J)[1] = 3.0 * x[1]^2
    return nothing
end

Jpat = spzeros(1, 1)
Jpat[1, 1] = 1.0
prob = Problem(r!, J!, Jpat)

state, work = initialize(prob, [0.5])
stats = solve!(state, prob, work)
```

## CAD‑inspired constraints

Fixed point `p1=(1,1)`, point `p2` constrained to the x‑axis, and distance `|p2-p1|=2`.

```julia
using SparseArrays
using SparseLNNS

function r!(out, x)
    x1, y1, x2, y2 = x
    out[1] = x1 - 1.0
    out[2] = y1 - 1.0
    out[3] = y2
    out[4] = (x2 - x1)^2 + (y2 - y1)^2 - 4.0
    return nothing
end

function J!(J, x)
    x1, y1, x2, y2 = x
    nz = nonzeros(J)
    nz[1] = 1.0
    nz[2] = 2.0 * (x1 - x2)
    nz[3] = 1.0
    nz[4] = 2.0 * (y1 - y2)
    nz[5] = 2.0 * (x2 - x1)
    nz[6] = 1.0
    nz[7] = 2.0 * (y2 - y1)
    return nothing
end

Jpat = spzeros(4, 4)
Jpat[1, 1] = 1.0
Jpat[4, 1] = 1.0
Jpat[2, 2] = 1.0
Jpat[4, 2] = 1.0
Jpat[4, 3] = 1.0
Jpat[3, 4] = 1.0
Jpat[4, 4] = 1.0

prob = Problem(r!, J!, Jpat)
state, work = initialize(prob, [0.8, 1.2, 2.6, 0.1])
stats = solve!(state, prob, work)
```

## Tips
- Build `Jpat` with `spzeros` and assign only the nonzeros you will fill.
- In `J!`, fill `nonzeros(J)` in a fixed order or write by index (`J[i,j] = ...`).
- Reuse `state` and `work` for repeated solves to avoid allocations.
