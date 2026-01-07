using Test
using SparseArrays
using NLLS

@testset "basic nonlinear solves" begin
    # Solve r(x)=x-1=0 -> x=1 (single residual, linear).
    r1!(out, x) = (out[1] = x[1] - 1.0)
    function J1!(J, x)
        nonzeros(J)[1] = 1.0
        return nothing
    end
    Jpat1 = spzeros(1, 1)
    Jpat1[1, 1] = 1.0
    prob1 = Problem(r1!, J1!, Jpat1)
    state1, work1 = initialize(prob1, [0.0])
    stats1 = solve!(state1, prob1, work1)
    @test stats1.status == :converged
    @test isapprox(state1.x[1], 1.0; atol=1e-6)

    # Solve r(x)=x^3-1=0 -> x=1 (nonlinear scalar).
    r2!(out, x) = (out[1] = x[1] * x[1] * x[1] - 1.0)
    function J2!(J, x)
        x1 = x[1]
        nonzeros(J)[1] = 3.0 * x1 * x1
        return nothing
    end
    Jpat2 = spzeros(1, 1)
    Jpat2[1, 1] = 1.0
    prob2 = Problem(r2!, J2!, Jpat2)
    state2, work2 = initialize(prob2, [0.5])
    stats2 = solve!(state2, prob2, work2)
    @test stats2.status == :converged
    @test isapprox(state2.x[1], 1.0; atol=1e-6)
end

@testset "multidimensional quadratics" begin
    # Solve x1^2=1, x2^2=4, x1*x2=2 -> (x1,x2)=(1,2).
    function r3!(out, x)
        x1 = x[1]
        x2 = x[2]
        out[1] = x1 * x1 - 1.0
        out[2] = x2 * x2 - 4.0
        out[3] = x1 * x2 - 2.0
        return nothing
    end
    function J3!(J, x)
        x1 = x[1]
        x2 = x[2]
        nz = nonzeros(J)
        nz[1] = 2.0 * x1
        nz[2] = x2
        nz[3] = 2.0 * x2
        nz[4] = x1
        return nothing
    end
    Jpat3 = spzeros(3, 2)
    Jpat3[1, 1] = 1.0
    Jpat3[3, 1] = 1.0
    Jpat3[2, 2] = 1.0
    Jpat3[3, 2] = 1.0
    prob3 = Problem(r3!, J3!, Jpat3)
    state3, work3 = initialize(prob3, [0.8, 1.7])
    stats3 = solve!(state3, prob3, work3)
    @test stats3.status == :converged
    @test isapprox(state3.x[1], 1.0; atol=1e-6)
    @test isapprox(state3.x[2], 2.0; atol=1e-6)
end

@testset "rosenbrock residuals" begin
    # Minimize Rosenbrock residuals: r=[1-x1, 10(x2-x1^2)] -> optimum (1,1).
    function r4!(out, x)
        x1 = x[1]
        x2 = x[2]
        out[1] = 1.0 - x1
        out[2] = 10.0 * (x2 - x1 * x1)
        return nothing
    end
    function J4!(J, x)
        x1 = x[1]
        nz = nonzeros(J)
        nz[1] = -1.0
        nz[2] = -20.0 * x1
        nz[3] = 10.0
        return nothing
    end
    Jpat4 = spzeros(2, 2)
    Jpat4[1, 1] = 1.0
    Jpat4[2, 1] = 1.0
    Jpat4[2, 2] = 1.0
    prob4 = Problem(r4!, J4!, Jpat4)
    state4, work4 = initialize(prob4, [-1.2, 1.0])
    stats4 = solve!(state4, prob4, work4)
    @test stats4.status == :converged
    @test isapprox(state4.x[1], 1.0; atol=1e-5)
    @test isapprox(state4.x[2], 1.0; atol=1e-5)
end

@testset "allocations" begin
    # Allocation check on the same linear residual r(x)=x-1 -> x=1.
    r!(out, x) = (out[1] = x[1] - 1.0)
    function J!(J, x)
        nonzeros(J)[1] = 1.0
        return nothing
    end
    Jpat = spzeros(1, 1)
    Jpat[1, 1] = 1.0
    prob = Problem(r!, J!, Jpat)
    state, work = initialize(prob, [0.0])
    solve!(state, prob, work) # warm up
    state2, work2 = initialize(prob, [0.0])
    alloc = @allocated solve!(state2, prob, work2)
    @test alloc <= 50_000
end
