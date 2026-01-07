module SparseLNNS

using LinearAlgebra
using SparseArrays
using SparseArrays: FixedSparseCSC, getcolptr, rowvals, nonzeros, nnz
using SparseArrays.SPQR

export Problem, Options, State, Stats, Workspace, initialize, solve!

"""
    Problem(r!, J!, jac_pattern)

Problem definition for sparse nonlinear least squares with a fixed Jacobian pattern.

- `r!(out, x)` writes residuals into `out` (length `m`).
- `J!(J, x)` fills Jacobian values in `J` using the sparsity of `jac_pattern`.
"""
struct Problem{T}
    r!::Function
    J!::Function
    m::Int
    n::Int
    jac_pattern::SparseMatrixCSC{T,Int}
end

function Problem(r!, J!, jac_pattern::SparseMatrixCSC{T,Int}) where {T<:AbstractFloat}
    m, n = size(jac_pattern)
    return Problem{T}(r!, J!, m, n, jac_pattern)
end

Problem(r!, J!, jac_pattern::SparseMatrixCSC) =
    Problem(r!, J!, SparseMatrixCSC{Float64,Int}(jac_pattern))

"""
    Options(; kwargs...)

Solver configuration: tolerances, iteration limits, damping bounds, and QR ordering.
"""
Base.@kwdef struct Options{T<:AbstractFloat}
    max_iters::Int = 50
    atol::T = T(1e-8)
    rtol::T = T(1e-8)
    gtol::T = T(1e-8)
    step_tol::T = T(1e-12)
    lambda_init::T = T(1e-3)
    lambda_min::T = T(1e-12)
    lambda_max::T = T(1e12)
    ordering::Int32 = SPQR.ORDERING_DEFAULT
end

"""
    Stats

Iteration statistics updated by `solve!`.
"""
mutable struct Stats{T<:AbstractFloat}
    iters::Int
    cost::T
    grad_norm::T
    step_norm::T
    status::Symbol
end

Stats{T}() where {T<:AbstractFloat} = Stats{T}(0, T(Inf), T(Inf), T(Inf), :init)

"""
    State

Mutable solver state: current parameters, damping value, and stats.
"""
mutable struct State{T<:AbstractFloat}
    x::Vector{T}
    lambda::T
    stats::Stats{T}
end

"""
    Workspace

Preallocated buffers used by the solver (residuals, Jacobian storage, steps, RHS).
"""
struct Workspace{T<:AbstractFloat}
    r::Vector{T}
    r_trial::Vector{T}
    J::FixedSparseCSC{T,Int}
    A::FixedSparseCSC{T,Int}
    g::Vector{T}
    step::Vector{T}
    x_trial::Vector{T}
    b_aug::Vector{T}
    diag_idx::Vector{Int}
end

function build_augmented_pattern(jac_pattern)
    # Build the fixed sparsity pattern for [J; sqrt(lambda) * I].
    T = eltype(jac_pattern)
    m, n = size(jac_pattern)
    colptrJ = getcolptr(jac_pattern)
    rowvalJ = rowvals(jac_pattern)
    nnzJ = nnz(jac_pattern)
    colptrA = Vector{Int}(undef, n + 1)
    rowvalA = Vector{Int}(undef, nnzJ + n)
    diag_idx = Vector{Int}(undef, n)
    idx = 1
    colptrA[1] = 1
    for col in 1:n
        colptrA[col] = idx
        for k in colptrJ[col]:(colptrJ[col + 1] - 1)
            rowvalA[idx] = rowvalJ[k]
            idx += 1
        end
        rowvalA[idx] = m + col
        diag_idx[col] = idx
        idx += 1
    end
    colptrA[n + 1] = idx
    nzvalA = zeros(T, nnzJ + n)
    A = FixedSparseCSC(m + n, n, colptrA, rowvalA, nzvalA)
    return A, diag_idx
end

"""
    initialize(problem, x0; options=nothing)

Allocate solver state and workspace for `problem` starting at `x0`.
Returns `(state, work)`; reuse both across solves to avoid allocations.
"""
function initialize(problem::Problem, x0; options=nothing)
    T = eltype(problem.jac_pattern)
    if options === nothing
        options = Options{T}()
    end
    length(x0) == problem.n || throw(DimensionMismatch("x0 length must be $(problem.n)"))
    x = Vector{T}(x0)
    r = zeros(T, problem.m)
    r_trial = similar(r)
    pattern = problem.jac_pattern
    J = FixedSparseCSC(problem.m, problem.n, getcolptr(pattern), rowvals(pattern), zeros(T, nnz(pattern)))
    A, diag_idx = build_augmented_pattern(pattern)
    g = zeros(T, problem.n)
    step = zeros(T, problem.n)
    x_trial = similar(x)
    b_aug = zeros(T, problem.m + problem.n)
    work = Workspace(r, r_trial, J, A, g, step, x_trial, b_aug, diag_idx)
    stats = Stats{T}()
    state = State{T}(x, options.lambda_init, stats)
    return state, work
end

function grad_norm_inf(g)
    # Infinity-norm of the gradient: max(|g_i|).
    T = eltype(g)
    maxval = zero(T)
    @inbounds for i in eachindex(g)
        val = abs(g[i])
        if val > maxval
            maxval = val
        end
    end
    return maxval
end

function update_augmented_values!(A, J, diag_idx, lambda)
    # Copy J into the top block of A and write sqrt(lambda) on the diagonal block.
    a_nz = nonzeros(A)
    j_nz = nonzeros(J)
    colptrJ = getcolptr(J)
    colptrA = getcolptr(A)
    n = size(J, 2)
    for col in 1:n
        aj = colptrA[col]
        jstart = colptrJ[col]
        jend = colptrJ[col + 1] - 1
        for k in jstart:jend
            a_nz[aj] = j_nz[k]
            aj += 1
        end
    end
    sqrt_lambda = sqrt(lambda)
    @inbounds for col in 1:n
        a_nz[diag_idx[col]] = sqrt_lambda
    end
    return nothing
end

function update_rhs!(b_aug, r, m, n)
    # Right-hand side for the augmented system [J; sqrt(lambda)I] * step = [-r; 0].
    @inbounds for i in 1:m
        b_aug[i] = -r[i]
    end
    @inbounds for i in 1:n
        b_aug[m + i] = zero(eltype(b_aug))
    end
    return nothing
end

function evaluate_current!(work::Workspace, problem::Problem, x)
    # Evaluate residual, Jacobian, cost, and gradient at the current x.
    T = eltype(work.r)
    problem.r!(work.r, x)
    problem.J!(work.J, x)
    cost = T(0.5) * dot(work.r, work.r)
    mul!(work.g, work.J', work.r)
    gnorm = grad_norm_inf(work.g)
    return cost, gnorm
end

function compute_step!(
        work::Workspace,
        problem::Problem,
        lambda,
        options)
    # Solve the damped least-squares subproblem via sparse QR.
    update_augmented_values!(work.A, work.J, work.diag_idx, lambda)
    update_rhs!(work.b_aug, work.r, problem.m, problem.n)
    F = qr(work.A; ordering=options.ordering)
    work.step .= F \ work.b_aug
    return nothing
end

function trial_step!(work::Workspace, x)
    # Form x_trial = x + step without allocating.
    @inbounds for i in eachindex(x)
        work.x_trial[i] = x[i] + work.step[i]
    end
    return nothing
end

function evaluate_trial!(work::Workspace, problem::Problem)
    # Compute trial residual and cost at x_trial.
    T = eltype(work.r_trial)
    problem.r!(work.r_trial, work.x_trial)
    return T(0.5) * dot(work.r_trial, work.r_trial)
end

function predicted_reduction(g, step, lambda)
    # Quadratic model decrease for LM step.
    T = eltype(step)
    pred = zero(T)
    @inbounds for i in eachindex(step)
        pred += step[i] * (lambda * step[i] - g[i])
    end
    return pred * T(0.5)
end

function accept_trial!(
        work::Workspace,
        problem::Problem,
        x,
        cost_trial)
    # Commit trial point and refresh Jacobian and gradient.
    copyto!(x, work.x_trial)
    copyto!(work.r, work.r_trial)
    problem.J!(work.J, x)
    mul!(work.g, work.J', work.r)
    gnorm = grad_norm_inf(work.g)
    return cost_trial, gnorm
end

function converged(cost, gnorm, rnorm0, options)
    rnorm = sqrt(2 * cost)
    return gnorm <= options.gtol || rnorm <= options.atol + options.rtol * rnorm0
end

"""
    solve!(state, problem, work; options=nothing)

Run the LM iterations in place. Updates `state.x` and `state.stats`, and returns `stats`.
"""
function solve!(state::State, problem::Problem, work::Workspace; options=nothing)
    if options === nothing
        options = Options{eltype(state.x)}()
    end
    T = eltype(state.x)
    x = state.x
    g = work.g
    step = work.step

    cost, gnorm = evaluate_current!(work, problem, x)
    rnorm0 = sqrt(T(2) * cost)
    lambda = state.lambda

    stats = state.stats
    stats.iters = 0
    stats.cost = cost
    stats.grad_norm = gnorm
    stats.step_norm = T(Inf)
    stats.status = :running

    for iter in 1:options.max_iters
        if converged(cost, gnorm, rnorm0, options)
            stats.status = :converged
            break
        end
        stats.iters = iter

        compute_step!(work, problem, lambda, options)

        step_norm = norm(step)
        stats.step_norm = step_norm
        if step_norm <= options.step_tol
            stats.status = :step_tol
            break
        end

        trial_step!(work, x)
        cost_trial = evaluate_trial!(work, problem)
        pred = predicted_reduction(g, step, lambda)

        if pred <= zero(T)
            lambda = min(lambda * T(2), options.lambda_max)
            continue
        end

        rho = (cost - cost_trial) / pred
        if cost_trial < cost
            cost, gnorm = accept_trial!(work, problem, x, cost_trial)

            if rho > T(0.75)
                lambda = max(lambda / T(2), options.lambda_min)
            elseif rho < T(0.25)
                lambda = min(lambda * T(2), options.lambda_max)
            end
        else
            lambda = min(lambda * T(2), options.lambda_max)
        end
    end

    if stats.status == :running
        stats.status = :max_iters
    end

    state.lambda = lambda
    stats.cost = cost
    stats.grad_norm = gnorm
    return stats
end

end # module SparseLNNS
