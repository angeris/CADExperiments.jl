using Test
using CADConstraints
using SparseLNNS: Options

const LOG_OPTIONS = Options(log=true)

@testset "points + lines constraints" begin
    # Three points with a horizontal and vertical line; p2 should land at (2, 0).
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

    stats = solve!(sk; options=LOG_OPTIONS)
    ix2, iy2 = 2 * (p2 - 1) + 1, 2 * (p2 - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ix2], 2.0; atol=1e-8)
    @test isapprox(sk.x[iy2], 0.0; atol=1e-8)
end

@testset "distance constraint" begin
    # Fix p1 at (0,0), enforce horizontal line, and set distance(p1,p2)=5.
    sk = Sketch()
    p1 = add_point!(sk, 0.0, 0.0)
    p2 = add_point!(sk, 4.0, 0.1)
    l1 = push!(sk, Line(p1, p2))

    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, Horizontal(l1))
    push!(sk, Distance(p1, p2, 5.0))

    stats = solve!(sk; options=LOG_OPTIONS)
    ix2, iy2 = 2 * (p2 - 1) + 1, 2 * (p2 - 1) + 2
    @test stats.status == :converged
    @test isapprox(abs(sk.x[ix2]), 5.0; atol=1e-8)
    @test isapprox(sk.x[iy2], 0.0; atol=1e-8)
end

@testset "circle diameter constraint" begin
    # Constrain a circle center to (0,0) and diameter to 10; lock orientation with a horizontal line.
    sk = Sketch()
    center = add_point!(sk, 0.2, -0.1)
    rim = add_point!(sk, 4.2, 1.0)
    c1 = push!(sk, Circle(center, rim))
    l1 = push!(sk, Line(center, rim))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, Horizontal(l1))
    push!(sk, Diameter(c1, 10.0))

    stats = solve!(sk; options=LOG_OPTIONS)
    ixr, iyr = 2 * (rim - 1) + 1, 2 * (rim - 1) + 2
    @test stats.status == :converged
    @test isapprox(abs(sk.x[ixr]), 5.0; atol=1e-8)
    @test isapprox(sk.x[iyr], 0.0; atol=1e-8)
end

@testset "value updates reuse problem" begin
    # Update a point in-place without structural edits; no allocations and no rebuild.
    sk = Sketch()
    p1 = add_point!(sk, -0.2, 0.1)
    p2 = add_point!(sk, 2.4, 0.8)
    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, FixedPoint(p2, 2.0, 1.0))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test stats.status == :converged
    old_problem = sk.problem

    alloc = @allocated set_point!(sk, p2, 4.0, -3.0)
    @test alloc == 0
    @test sk.structure_dirty == false
    @test sk.value_dirty == true

    stats = solve!(sk; options=LOG_OPTIONS)
    @test stats.status == :converged
    @test sk.problem === old_problem
end

@testset "complex sketch constraints" begin
    # Rectangle with duplicated corner points tied by coincident + parallel constraints.
    sk = Sketch()
    p1a = add_point!(sk, -0.2, 0.1)
    p1b = add_point!(sk, 0.1, -0.1)
    p2a = add_point!(sk, 4.1, 0.2)
    p2b = add_point!(sk, 3.9, -0.2)
    p3a = add_point!(sk, 4.2, 3.1)
    p3b = add_point!(sk, 3.8, 2.9)
    p4a = add_point!(sk, -0.1, 3.2)
    p4b = add_point!(sk, 0.2, 2.8)

    l1 = push!(sk, Line(p1a, p2a)) # bottom
    l2 = push!(sk, Line(p2b, p3a)) # right
    l3 = push!(sk, Line(p3b, p4a)) # top
    l4 = push!(sk, Line(p4b, p1b)) # left

    push!(sk, Coincident(p1a, p1b))
    push!(sk, Coincident(p2a, p2b))
    push!(sk, Coincident(p3a, p3b))
    push!(sk, Coincident(p4a, p4b))

    push!(sk, FixedPoint(p1a, 0.0, 0.0))
    push!(sk, FixedPoint(p2a, 4.0, 0.0))
    push!(sk, FixedPoint(p4a, 0.0, 3.0))

    push!(sk, Horizontal(l1))
    push!(sk, Vertical(l2))
    push!(sk, Horizontal(l3))
    push!(sk, Vertical(l4))
    push!(sk, Parallel(l1, l3))
    push!(sk, Parallel(l2, l4))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test stats.status == :converged

    ix3a, iy3a = 2 * (p3a - 1) + 1, 2 * (p3a - 1) + 2
    @test isapprox(sk.x[ix3a], 4.0; atol=1e-6)
    @test isapprox(sk.x[iy3a], 3.0; atol=1e-6)

    for (pa, pb) in ((p1a, p1b), (p2a, p2b), (p3a, p3b), (p4a, p4b))
        ix1, iy1 = 2 * (pa - 1) + 1, 2 * (pa - 1) + 2
        ix2, iy2 = 2 * (pb - 1) + 1, 2 * (pb - 1) + 2
        @test isapprox(sk.x[ix1], sk.x[ix2]; atol=1e-6)
        @test isapprox(sk.x[iy1], sk.x[iy2]; atol=1e-6)
    end
end

@testset "circle + line + point interplay" begin
    # Circle diameter + vertical line to a fixed point should place the rim at (5,0).
    sk = Sketch()
    center = add_point!(sk, 0.1, 0.2)
    rim = add_point!(sk, 4.2, 0.7)
    anchor = add_point!(sk, 4.9, 0.0)

    c1 = push!(sk, Circle(center, rim))
    l1 = push!(sk, Line(rim, anchor))
    l2 = push!(sk, Line(center, rim))
    l3 = push!(sk, Line(center, anchor))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, Distance(center, anchor, 5.0))
    push!(sk, Horizontal(l3))
    push!(sk, Diameter(c1, 10.0))
    push!(sk, Vertical(l1))
    push!(sk, Horizontal(l2))

    stats = solve!(sk)
    ixr, iyr = 2 * (rim - 1) + 1, 2 * (rim - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ixr], 5.0; atol=1e-6)
    @test isapprox(sk.x[iyr], 0.0; atol=1e-6)
end
