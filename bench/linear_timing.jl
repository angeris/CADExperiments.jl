using LinearAlgebra
using Random
using SparseArrays
using SparseLNNS

function linear_problem(n::Int, density::Float64; seed::Int=0)
    Random.seed!(seed)
    A = sprand(n, n, density)
    x_true = randn(n)
    b = A * x_true
    nzA = nonzeros(A)

    function r!(out, x)
        mul!(out, A, x)
        @inbounds for i in eachindex(out)
            out[i] -= b[i]
        end
        return nothing
    end

    function J!(J, x)
        copyto!(nonzeros(J), nzA)
        return nothing
    end

    prob = Problem(r!, J!, A)
    return prob
end

function run_case(n::Int, density::Float64)
    prob = linear_problem(n, density; seed=1)
    options = Options{Float64}(lambda_init=0.0, max_iters=5)
    state, work = initialize(prob, zeros(n); options=options)
    solve!(state, prob, work; options=options) # warm up
    fill!(state.x, 0.0)
    state.lambda = options.lambda_init
    t = @elapsed stats = solve!(state, prob, work; options=options)
    return (stats=stats, time=t, nnz=nnz(prob.jac_pattern))
end

n = 1000
for density in (0.0001, 0.001)
    result = run_case(n, density)
    println("n=$n density=$(density) nnz=$(result.nnz) iters=$(result.stats.iters) time=$(result.time) sec")
end
