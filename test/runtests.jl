using Test
using SparseArrays
using SparseLNNS

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

@testset "cad-inspired: axis + distance" begin
    # Fixed point (x1,y1)=(1,1), point2 on x-axis, and distance=2 to point1.
    function r5!(out, x)
        x1 = x[1]
        y1 = x[2]
        x2 = x[3]
        y2 = x[4]
        out[1] = x1 - 1.0
        out[2] = y1 - 1.0
        out[3] = y2
        out[4] = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) - 4.0
        return nothing
    end
    function J5!(J, x)
        x1 = x[1]
        y1 = x[2]
        x2 = x[3]
        y2 = x[4]
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
    Jpat5 = spzeros(4, 4)
    Jpat5[1, 1] = 1.0
    Jpat5[4, 1] = 1.0
    Jpat5[2, 2] = 1.0
    Jpat5[4, 2] = 1.0
    Jpat5[4, 3] = 1.0
    Jpat5[3, 4] = 1.0
    Jpat5[4, 4] = 1.0
    prob5 = Problem(r5!, J5!, Jpat5)
    state5, work5 = initialize(prob5, [0.8, 1.2, 2.6, 0.1])
    stats5 = solve!(state5, prob5, work5)
    x2_expected = 1.0 + sqrt(3.0)
    @test stats5.status == :converged
    @test isapprox(state5.x[1], 1.0; atol=1e-5)
    @test isapprox(state5.x[2], 1.0; atol=1e-5)
    @test isapprox(state5.x[3], x2_expected; atol=1e-5)
    @test isapprox(state5.x[4], 0.0; atol=1e-5)
end

@testset "cad-inspired: parallel lines" begin
    # Line1 fixed to x-axis; Line2 anchored at y=1, parallel and length=2 -> p4=(2,1).
    function r6!(out, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        x3 = x[5]; y3 = x[6]
        x4 = x[7]; y4 = x[8]
        out[1] = x1
        out[2] = y1
        out[3] = x2 - 2.0
        out[4] = y2
        out[5] = x3
        out[6] = y3 - 1.0
        out[7] = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3) - 4.0
        out[8] = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        return nothing
    end
    function J6!(J, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        x3 = x[5]; y3 = x[6]
        x4 = x[7]; y4 = x[8]
        dx12 = x2 - x1
        dy12 = y2 - y1
        dx34 = x4 - x3
        dy34 = y4 - y3
        fill!(nonzeros(J), 0.0)
        J[1, 1] = 1.0
        J[2, 2] = 1.0
        J[3, 3] = 1.0
        J[4, 4] = 1.0
        J[5, 5] = 1.0
        J[6, 6] = 1.0
        J[7, 5] = -2.0 * dx34
        J[7, 6] = -2.0 * dy34
        J[7, 7] = 2.0 * dx34
        J[7, 8] = 2.0 * dy34
        J[8, 1] = -dy34
        J[8, 2] = dx34
        J[8, 3] = dy34
        J[8, 4] = -dx34
        J[8, 5] = dy12
        J[8, 6] = -dx12
        J[8, 7] = -dy12
        J[8, 8] = dx12
        return nothing
    end
    Jpat6 = spzeros(8, 8)
    Jpat6[1, 1] = 1.0
    Jpat6[2, 2] = 1.0
    Jpat6[3, 3] = 1.0
    Jpat6[4, 4] = 1.0
    Jpat6[5, 5] = 1.0
    Jpat6[6, 6] = 1.0
    Jpat6[7, 5] = 1.0
    Jpat6[7, 6] = 1.0
    Jpat6[7, 7] = 1.0
    Jpat6[7, 8] = 1.0
    Jpat6[8, 1] = 1.0
    Jpat6[8, 2] = 1.0
    Jpat6[8, 3] = 1.0
    Jpat6[8, 4] = 1.0
    Jpat6[8, 5] = 1.0
    Jpat6[8, 6] = 1.0
    Jpat6[8, 7] = 1.0
    Jpat6[8, 8] = 1.0
    prob6 = Problem(r6!, J6!, Jpat6)
    state6, work6 = initialize(prob6, [0.1, -0.1, 2.1, 0.2, -0.1, 1.1, 1.9, 1.2])
    stats6 = solve!(state6, prob6, work6)
    @test stats6.status == :converged
    @test isapprox(state6.x[1], 0.0; atol=1e-5)
    @test isapprox(state6.x[2], 0.0; atol=1e-5)
    @test isapprox(state6.x[3], 2.0; atol=1e-5)
    @test isapprox(state6.x[4], 0.0; atol=1e-5)
    @test isapprox(state6.x[5], 0.0; atol=1e-5)
    @test isapprox(state6.x[6], 1.0; atol=1e-5)
    @test isapprox(state6.x[7], 2.0; atol=1e-5)
    @test isapprox(state6.x[8], 1.0; atol=1e-5)
end

@testset "cad-inspired: perpendicular lines" begin
    # Line1 fixed to x-axis, Line2 anchored at (1,0), perpendicular with y4=2 -> p4=(1,2).
    function r7!(out, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        x3 = x[5]; y3 = x[6]
        x4 = x[7]; y4 = x[8]
        out[1] = x1
        out[2] = y1
        out[3] = x2 - 2.0
        out[4] = y2
        out[5] = x3 - 1.0
        out[6] = y3
        out[7] = y4 - 2.0
        out[8] = (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)
        return nothing
    end
    function J7!(J, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        x3 = x[5]; y3 = x[6]
        x4 = x[7]; y4 = x[8]
        dx12 = x2 - x1
        dy12 = y2 - y1
        dx34 = x4 - x3
        dy34 = y4 - y3
        fill!(nonzeros(J), 0.0)
        J[1, 1] = 1.0
        J[2, 2] = 1.0
        J[3, 3] = 1.0
        J[4, 4] = 1.0
        J[5, 5] = 1.0
        J[6, 6] = 1.0
        J[7, 8] = 1.0
        J[8, 1] = -dx34
        J[8, 2] = -dy34
        J[8, 3] = dx34
        J[8, 4] = dy34
        J[8, 5] = -dx12
        J[8, 6] = -dy12
        J[8, 7] = dx12
        J[8, 8] = dy12
        return nothing
    end
    Jpat7 = spzeros(8, 8)
    Jpat7[1, 1] = 1.0
    Jpat7[2, 2] = 1.0
    Jpat7[3, 3] = 1.0
    Jpat7[4, 4] = 1.0
    Jpat7[5, 5] = 1.0
    Jpat7[6, 6] = 1.0
    Jpat7[7, 8] = 1.0
    Jpat7[8, 1] = 1.0
    Jpat7[8, 2] = 1.0
    Jpat7[8, 3] = 1.0
    Jpat7[8, 4] = 1.0
    Jpat7[8, 5] = 1.0
    Jpat7[8, 6] = 1.0
    Jpat7[8, 7] = 1.0
    Jpat7[8, 8] = 1.0
    prob7 = Problem(r7!, J7!, Jpat7)
    state7, work7 = initialize(prob7, [0.1, -0.1, 2.1, 0.1, 1.1, 0.1, 0.9, 2.1])
    stats7 = solve!(state7, prob7, work7)
    @test stats7.status == :converged
    @test isapprox(state7.x[7], 1.0; atol=1e-5)
    @test isapprox(state7.x[8], 2.0; atol=1e-5)
end

@testset "cad-inspired: tangent circle" begin
    # Circle tangent to x-axis and passing through (2,1) -> center (1,1), r=1.
    function r8!(out, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        cx = x[5]; cy = x[6]
        r = x[7]
        x3 = x[8]; y3 = x[9]
        out[1] = x1
        out[2] = y1
        out[3] = x2 - 2.0
        out[4] = y2
        out[5] = cx - 1.0
        out[6] = cy - r
        out[7] = (x3 - cx) * (x3 - cx) + (y3 - cy) * (y3 - cy) - r * r
        out[8] = x3 - 2.0
        out[9] = y3 - 1.0
        return nothing
    end
    function J8!(J, x)
        cx = x[5]; cy = x[6]
        r = x[7]
        x3 = x[8]; y3 = x[9]
        dx = x3 - cx
        dy = y3 - cy
        fill!(nonzeros(J), 0.0)
        J[1, 1] = 1.0
        J[2, 2] = 1.0
        J[3, 3] = 1.0
        J[4, 4] = 1.0
        J[5, 5] = 1.0
        J[6, 6] = 1.0
        J[6, 7] = -1.0
        J[7, 5] = -2.0 * dx
        J[7, 6] = -2.0 * dy
        J[7, 7] = -2.0 * r
        J[7, 8] = 2.0 * dx
        J[7, 9] = 2.0 * dy
        J[8, 8] = 1.0
        J[9, 9] = 1.0
        return nothing
    end
    Jpat8 = spzeros(9, 9)
    Jpat8[1, 1] = 1.0
    Jpat8[2, 2] = 1.0
    Jpat8[3, 3] = 1.0
    Jpat8[4, 4] = 1.0
    Jpat8[5, 5] = 1.0
    Jpat8[6, 6] = 1.0
    Jpat8[6, 7] = 1.0
    Jpat8[7, 5] = 1.0
    Jpat8[7, 6] = 1.0
    Jpat8[7, 7] = 1.0
    Jpat8[7, 8] = 1.0
    Jpat8[7, 9] = 1.0
    Jpat8[8, 8] = 1.0
    Jpat8[9, 9] = 1.0
    prob8 = Problem(r8!, J8!, Jpat8)
    state8, work8 = initialize(prob8, [0.1, -0.1, 2.1, 0.1, 0.9, 0.8, 0.9, 2.1, 0.9])
    stats8 = solve!(state8, prob8, work8)
    @test stats8.status == :converged
    @test isapprox(state8.x[5], 1.0; atol=1e-5)
    @test isapprox(state8.x[6], 1.0; atol=1e-5)
    @test isapprox(state8.x[7], 1.0; atol=1e-5)
end

@testset "cad-inspired: complex constraints" begin
    # Coincident points, parallel lines, and tangent circle -> p4=(2,1), center (1,1), r=1.
    function r9!(out, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        x3 = x[5]; y3 = x[6]
        x4 = x[7]; y4 = x[8]
        x5 = x[9]; y5 = x[10]
        x6 = x[11]; y6 = x[12]
        cx = x[13]; cy = x[14]
        r = x[15]
        out[1] = x1
        out[2] = y1
        out[3] = x2 - 2.0
        out[4] = y2
        out[5] = x3
        out[6] = y3 - 1.0
        out[7] = x5 - x1
        out[8] = y5 - y1
        out[9] = x6 - x3
        out[10] = y6 - y3
        out[11] = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        out[12] = (x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3) - 4.0
        out[13] = cx - 1.0
        out[14] = cy - 1.0
        out[15] = cy - r
        out[16] = (x4 - cx) * (x4 - cx) + (y4 - cy) * (y4 - cy) - r * r
        return nothing
    end
    function J9!(J, x)
        x1 = x[1]; y1 = x[2]
        x2 = x[3]; y2 = x[4]
        x3 = x[5]; y3 = x[6]
        x4 = x[7]; y4 = x[8]
        x5 = x[9]; y5 = x[10]
        x6 = x[11]; y6 = x[12]
        cx = x[13]; cy = x[14]
        r = x[15]
        dx12 = x2 - x1
        dy12 = y2 - y1
        dx34 = x4 - x3
        dy34 = y4 - y3
        dx4 = x4 - cx
        dy4 = y4 - cy
        fill!(nonzeros(J), 0.0)
        J[1, 1] = 1.0
        J[2, 2] = 1.0
        J[3, 3] = 1.0
        J[4, 4] = 1.0
        J[5, 5] = 1.0
        J[6, 6] = 1.0
        J[7, 1] = -1.0
        J[7, 9] = 1.0
        J[8, 2] = -1.0
        J[8, 10] = 1.0
        J[9, 5] = -1.0
        J[9, 11] = 1.0
        J[10, 6] = -1.0
        J[10, 12] = 1.0
        J[11, 1] = -dy34
        J[11, 2] = dx34
        J[11, 3] = dy34
        J[11, 4] = -dx34
        J[11, 5] = dy12
        J[11, 6] = -dx12
        J[11, 7] = -dy12
        J[11, 8] = dx12
        J[12, 5] = -2.0 * dx34
        J[12, 6] = -2.0 * dy34
        J[12, 7] = 2.0 * dx34
        J[12, 8] = 2.0 * dy34
        J[13, 13] = 1.0
        J[14, 14] = 1.0
        J[15, 14] = 1.0
        J[15, 15] = -1.0
        J[16, 7] = 2.0 * dx4
        J[16, 8] = 2.0 * dy4
        J[16, 13] = -2.0 * dx4
        J[16, 14] = -2.0 * dy4
        J[16, 15] = -2.0 * r
        return nothing
    end
    Jpat9 = spzeros(16, 15)
    Jpat9[1, 1] = 1.0
    Jpat9[2, 2] = 1.0
    Jpat9[3, 3] = 1.0
    Jpat9[4, 4] = 1.0
    Jpat9[5, 5] = 1.0
    Jpat9[6, 6] = 1.0
    Jpat9[7, 1] = 1.0
    Jpat9[7, 9] = 1.0
    Jpat9[8, 2] = 1.0
    Jpat9[8, 10] = 1.0
    Jpat9[9, 5] = 1.0
    Jpat9[9, 11] = 1.0
    Jpat9[10, 6] = 1.0
    Jpat9[10, 12] = 1.0
    Jpat9[11, 1] = 1.0
    Jpat9[11, 2] = 1.0
    Jpat9[11, 3] = 1.0
    Jpat9[11, 4] = 1.0
    Jpat9[11, 5] = 1.0
    Jpat9[11, 6] = 1.0
    Jpat9[11, 7] = 1.0
    Jpat9[11, 8] = 1.0
    Jpat9[12, 5] = 1.0
    Jpat9[12, 6] = 1.0
    Jpat9[12, 7] = 1.0
    Jpat9[12, 8] = 1.0
    Jpat9[13, 13] = 1.0
    Jpat9[14, 14] = 1.0
    Jpat9[15, 14] = 1.0
    Jpat9[15, 15] = 1.0
    Jpat9[16, 7] = 1.0
    Jpat9[16, 8] = 1.0
    Jpat9[16, 13] = 1.0
    Jpat9[16, 14] = 1.0
    Jpat9[16, 15] = 1.0
    prob9 = Problem(r9!, J9!, Jpat9)
    x0 = [0.1, -0.1, 2.1, 0.2, -0.1, 1.1, 1.9, 0.9, 0.2, 0.1, -0.1, 1.2, 0.9, 1.2, 1.1]
    state9, work9 = initialize(prob9, x0)
    stats9 = solve!(state9, prob9, work9)
    @test stats9.status == :converged
    @test isapprox(state9.x[1], 0.0; atol=1e-5)
    @test isapprox(state9.x[2], 0.0; atol=1e-5)
    @test isapprox(state9.x[3], 2.0; atol=1e-5)
    @test isapprox(state9.x[4], 0.0; atol=1e-5)
    @test isapprox(state9.x[5], 0.0; atol=1e-5)
    @test isapprox(state9.x[6], 1.0; atol=1e-5)
    @test isapprox(state9.x[7], 2.0; atol=1e-5)
    @test isapprox(state9.x[8], 1.0; atol=1e-5)
    @test isapprox(state9.x[9], 0.0; atol=1e-5)
    @test isapprox(state9.x[10], 0.0; atol=1e-5)
    @test isapprox(state9.x[11], 0.0; atol=1e-5)
    @test isapprox(state9.x[12], 1.0; atol=1e-5)
    @test isapprox(state9.x[13], 1.0; atol=1e-5)
    @test isapprox(state9.x[14], 1.0; atol=1e-5)
    @test isapprox(state9.x[15], 1.0; atol=1e-5)
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
