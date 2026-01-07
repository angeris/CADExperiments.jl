module SparseLNNS

using LinearAlgebra
using Printf
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
struct Problem
    r!::Function
    J!::Function
    m::Int
    n::Int
    jac_pattern::SparseMatrixCSC{Float64,Int}
end

function Problem(r!, J!, jac_pattern::SparseMatrixCSC{Float64,Int})
    m, n = size(jac_pattern)
    return Problem(r!, J!, m, n, jac_pattern)
end

"""
    Options(; kwargs...)

Solver configuration: tolerances, iteration limits, damping bounds, QR ordering, and logging.
"""
Base.@kwdef struct Options
    max_iters::Int = 50
    atol::Float64 = 1e-8
    rtol::Float64 = 1e-8
    gtol::Float64 = 1e-8
    step_tol::Float64 = 1e-12
    lambda_init::Float64 = 1e-3
    lambda_min::Float64 = 1e-12
    lambda_max::Float64 = 1e12
    ordering::Int32 = SPQR.ORDERING_DEFAULT
    log::Bool = false
    log_io::IO = stdout
    log_every::Int = 1
end

"""
    Stats

Iteration statistics updated by `solve!`.
"""
mutable struct Stats
    iters::Int
    cost::Float64
    grad_norm::Float64
    step_norm::Float64
    status::Symbol
end

Stats() = Stats(0, Inf, Inf, Inf, :init)

"""
    State

Mutable solver state: current parameters, damping value, and stats.
"""
mutable struct State
    x::Vector{Float64}
    lambda::Float64
    stats::Stats
end

"""
    Workspace

Preallocated buffers used by the solver (residuals, Jacobian storage, steps, RHS).
"""
struct Workspace
    r::Vector{Float64}
    r_trial::Vector{Float64}
    J::FixedSparseCSC{Float64,Int}
    A::FixedSparseCSC{Float64,Int}
    g::Vector{Float64}
    step::Vector{Float64}
    x_trial::Vector{Float64}
    b_aug::Vector{Float64}
    diag_idx::Vector{Int}
end

function build_augmented_pattern(jac_pattern::SparseMatrixCSC{Float64,Int})
    # Build the fixed sparsity pattern for [J; sqrt(lambda) * I].
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
    nzvalA = zeros(Float64, nnzJ + n)
    A = FixedSparseCSC(m + n, n, colptrA, rowvalA, nzvalA)
    return A, diag_idx
end

"""
    initialize(problem, x0; options=Options())

Allocate solver state and workspace for `problem` starting at `x0`.
Returns `(state, work)`; reuse both across solves to avoid allocations.
"""
function initialize(problem::Problem, x0; options=Options())
    length(x0) == problem.n || throw(DimensionMismatch("x0 length must be $(problem.n)"))
    x = Vector{Float64}(undef, problem.n)
    @inbounds for i in eachindex(x)
        x[i] = x0[i]
    end
    r = zeros(Float64, problem.m)
    r_trial = similar(r)
    pattern = problem.jac_pattern
    J = FixedSparseCSC(problem.m, problem.n, getcolptr(pattern), rowvals(pattern), zeros(Float64, nnz(pattern)))
    A, diag_idx = build_augmented_pattern(pattern)
    g = zeros(Float64, problem.n)
    step = zeros(Float64, problem.n)
    x_trial = similar(x)
    b_aug = zeros(Float64, problem.m + problem.n)
    work = Workspace(r, r_trial, J, A, g, step, x_trial, b_aug, diag_idx)
    stats = Stats()
    state = State(x, options.lambda_init, stats)
    return state, work
end

function grad_norm_inf(g)
    # Infinity-norm of the gradient: max(|g_i|).
    maxval = 0.0
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
        b_aug[m + i] = 0.0
    end
    return nothing
end

function evaluate_current!(work, problem, x)
    # Evaluate residual, Jacobian, cost, and gradient at the current x.
    problem.r!(work.r, x)
    problem.J!(work.J, x)
    cost = 0.5 * dot(work.r, work.r)
    mul!(work.g, work.J', work.r)
    gnorm = grad_norm_inf(work.g)
    return cost, gnorm
end

function compute_step!(
        work,
        problem,
        lambda,
        options)
    # Solve the damped least-squares subproblem via sparse QR.
    update_augmented_values!(work.A, work.J, work.diag_idx, lambda)
    update_rhs!(work.b_aug, work.r, problem.m, problem.n)
    F = qr(work.A; ordering=options.ordering)
    work.step .= F \ work.b_aug
    return nothing
end

function trial_step!(work, x)
    # Form x_trial = x + step without allocating.
    @inbounds for i in eachindex(x)
        work.x_trial[i] = x[i] + work.step[i]
    end
    return nothing
end

function evaluate_trial!(work, problem)
    # Compute trial residual and cost at x_trial.
    problem.r!(work.r_trial, work.x_trial)
    return 0.5 * dot(work.r_trial, work.r_trial)
end

function predicted_reduction(g, step, lambda)
    # Quadratic model decrease for LM step.
    pred = 0.0
    @inbounds for i in eachindex(step)
        pred += step[i] * (lambda * step[i] - g[i])
    end
    return pred * 0.5
end

function accept_trial!(
        work,
        problem,
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

function log_header(io)
    @printf(io, " iter |      cost |     rnorm |     gnorm |     step |   lambda |   rho\n")
    @printf(io, "------+-----------+-----------+-----------+----------+----------+-------\n")
    return nothing
end

function log_row(io, iter, cost, rnorm, gnorm, step_norm, lambda, rho)
    @printf(
        io,
        "%5d | %9.2e | %9.2e | %9.2e | %8.2e | %8.2e | %5.2f\n",
        iter,
        cost,
        rnorm,
        gnorm,
        step_norm,
        lambda,
        rho,
    )
    return nothing
end

"""
    solve!(state, problem, work; options=Options())

Run the LM iterations in place. Updates `state.x` and `state.stats`, and returns `stats`.
"""
function solve!(state::State, problem::Problem, work::Workspace;
        options=Options())
    x = state.x
    g = work.g
    step = work.step

    cost, gnorm = evaluate_current!(work, problem, x)
    rnorm0 = sqrt(2 * cost)
    lambda = state.lambda
    log_io = options.log_io
    log_enabled = options.log
    if log_enabled
        log_header(log_io)
        log_row(log_io, 0, cost, rnorm0, gnorm, 0.0, lambda, NaN)
    end

    stats = state.stats
    stats.iters = 0
    stats.cost = cost
    stats.grad_norm = gnorm
    stats.step_norm = Inf
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
            if log_enabled && iter % options.log_every == 0
                log_row(log_io, iter, cost, sqrt(2 * cost), gnorm, step_norm, lambda, NaN)
            end
            break
        end

        trial_step!(work, x)
        cost_trial = evaluate_trial!(work, problem)
        pred = predicted_reduction(g, step, lambda)

        rho = NaN
        if pred <= 0
            lambda = min(lambda * 2, options.lambda_max)
        else
            rho = (cost - cost_trial) / pred
            if cost_trial < cost
                cost, gnorm = accept_trial!(work, problem, x, cost_trial)

                if rho > 0.75
                    lambda = max(lambda / 2, options.lambda_min)
                elseif rho < 0.25
                    lambda = min(lambda * 2, options.lambda_max)
                end
            else
                lambda = min(lambda * 2, options.lambda_max)
            end
        end

        if log_enabled && iter % options.log_every == 0
            log_row(log_io, iter, cost, sqrt(2 * cost), gnorm, step_norm, lambda, rho)
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
