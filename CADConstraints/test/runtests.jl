using Test
using CADConstraints

@testset "points + lines constraints" begin
    sk = Sketch()
    p1 = add_point!(sk, 0.0, 0.0)
    p2 = add_point!(sk, 0.4, 0.2)
    p3 = add_point!(sk, 2.0, 1.0)

    l1 = add_line!(sk, p1, p2)
    l2 = add_line!(sk, p2, p3)

    add_fixed_point!(sk, p1, 0.0, 0.0)
    add_fixed_point!(sk, p3, 2.0, 1.0)
    add_horizontal!(sk, l1)
    add_vertical!(sk, l2)

    stats = solve!(sk)
    ix2, iy2 = 2 * (p2 - 1) + 1, 2 * (p2 - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ix2], 2.0; atol=1e-5)
    @test isapprox(sk.x[iy2], 0.0; atol=1e-5)
end
