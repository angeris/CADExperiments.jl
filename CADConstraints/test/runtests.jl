using Test
using CADConstraints

@testset "points + lines constraints" begin
    sk = Sketch()
    p1 = add_point!(sk, 0.0, 0.0)
    p2 = add_point!(sk, 0.4, 0.2)
    p3 = add_point!(sk, 2.0, 1.0)

    l1 = push!(sk, Line(p1, p2))
    l2 = push!(sk, Line(p2, p3))

    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, FixedPoint(p3, 2.0, 1.0))
    push!(sk, Horizontal(l1))
    push!(sk, Vertical(l2))

    stats = solve!(sk)
    ix2, iy2 = 2 * (p2 - 1) + 1, 2 * (p2 - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ix2], 2.0; atol=1e-5)
    @test isapprox(sk.x[iy2], 0.0; atol=1e-5)
end
