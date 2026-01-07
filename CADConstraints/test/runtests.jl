using Test
using CADConstraints
using SparseLNNS: Options

const LOG_OPTIONS = Options(log=true)
const LOG_TIGHT_OPTIONS = Options(log=true, atol=1e-10, rtol=1e-10, gtol=1e-10, max_iters=100)

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

    stats = solve!(sk; options=LOG_TIGHT_OPTIONS)
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

@testset "circle coincident constraint" begin
    # Point on a radius-2 circle constrained to the vertical line x=0.
    sk = Sketch()
    center = add_point!(sk, 0.0, 0.0)
    rim = add_point!(sk, 0.0, 2.0)
    p1 = add_point!(sk, 0.2, 1.6)
    anchor = add_point!(sk, 0.0, 0.0)

    c1 = push!(sk, Circle(center, rim))
    l1 = push!(sk, Line(p1, anchor))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, FixedPoint(rim, 0.0, 2.0))
    push!(sk, FixedPoint(anchor, 0.0, 0.0))
    push!(sk, Vertical(l1))
    push!(sk, CircleCoincident(c1, p1))

    stats = solve!(sk; options=LOG_TIGHT_OPTIONS)
    ix1, iy1 = 2 * (p1 - 1) + 1, 2 * (p1 - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ix1], 0.0; atol=1e-6)
    @test isapprox(sk.x[iy1], 2.0; atol=1e-6)
end

@testset "tangent via normal + circle coincident" begin
    # Tangency can be built from a normal line + point-on-circle.
    sk = Sketch()
    center = add_point!(sk, 0.0, 0.0)
    rim = add_point!(sk, 0.0, 2.0)
    contact = add_point!(sk, 0.2, 1.8)
    normal_aux = add_point!(sk, 0.2, 3.2)
    tangent_aux = add_point!(sk, 3.2, 1.0)
    anchor = add_point!(sk, 3.0, 0.0)

    c1 = push!(sk, Circle(center, rim))
    l1 = push!(sk, Line(contact, normal_aux))
    l2 = push!(sk, Line(contact, tangent_aux))
    l3 = push!(sk, Line(tangent_aux, anchor))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, FixedPoint(rim, 0.0, 2.0))
    push!(sk, FixedPoint(anchor, 3.0, 0.0))
    push!(sk, CircleCoincident(c1, contact))
    push!(sk, Normal(c1, l1))
    push!(sk, Vertical(l1))
    push!(sk, Horizontal(l2))
    push!(sk, Vertical(l3))

    stats = solve!(sk; options=LOG_TIGHT_OPTIONS)
    ixc, iyc = 2 * (contact - 1) + 1, 2 * (contact - 1) + 2
    ixt, iyt = 2 * (tangent_aux - 1) + 1, 2 * (tangent_aux - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ixc], 0.0; atol=1e-6)
    @test isapprox(sk.x[iyc], 2.0; atol=1e-6)
    @test isapprox(sk.x[ixt], 3.0; atol=1e-6)
    @test isapprox(sk.x[iyt], 2.0; atol=1e-6)
end

@testset "circle normal constraint" begin
    # A line constrained normal to a circle must pass through the center (p2 -> (0,0)).
    sk = Sketch()
    center = add_point!(sk, 0.0, 0.0)
    rim = add_point!(sk, 1.0, 0.0)
    p1 = add_point!(sk, 2.0, 1.0)
    p2 = add_point!(sk, 0.0, 2.0)
    pfix = add_point!(sk, 0.0, -1.0)

    c1 = push!(sk, Circle(center, rim))
    l1 = push!(sk, Line(p1, p2))
    l2 = push!(sk, Line(p2, pfix))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, FixedPoint(rim, 1.0, 0.0))
    push!(sk, FixedPoint(p1, 2.0, 1.0))
    push!(sk, FixedPoint(pfix, 0.0, -1.0))
    push!(sk, Vertical(l2))
    push!(sk, Normal(c1, l1))

    stats = solve!(sk; options=LOG_OPTIONS)
    ix2, iy2 = 2 * (p2 - 1) + 1, 2 * (p2 - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ix2], 0.0; atol=1e-6)
    @test isapprox(sk.x[iy2], 0.0; atol=1e-6)
end

@testset "arc shape endpoints" begin
    # Arc endpoints constrained to a radius-2 circle: start (2,0) and end (0,2).
    sk = Sketch()
    center = add_point!(sk, 0.1, -0.1)
    start = add_point!(sk, 1.9, 0.1)
    finish = add_point!(sk, -0.2, 1.8)

    arc = push!(sk, Arc(center, start, finish))
    l1 = push!(sk, Line(center, start))
    l2 = push!(sk, Line(center, finish))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, Horizontal(l1))
    push!(sk, Vertical(l2))
    push!(sk, Distance(center, start, 2.0))
    push!(sk, Distance(center, finish, 2.0))

    stats = solve!(sk; options=LOG_OPTIONS)
    ix1, iy1 = 2 * (start - 1) + 1, 2 * (start - 1) + 2
    ix2, iy2 = 2 * (finish - 1) + 1, 2 * (finish - 1) + 2
    @test stats.status == :converged
    @test sk.arcs[arc] == Arc(center, start, finish)
    @test isapprox(sk.x[ix1], 2.0; atol=1e-6)
    @test isapprox(sk.x[iy1], 0.0; atol=1e-6)
    @test isapprox(sk.x[ix2], 0.0; atol=1e-6)
    @test isapprox(sk.x[iy2], 2.0; atol=1e-6)
end

@testset "wiggle warm start" begin
    # Perturb a fixed anchor and re-solve to test warm-started updates.
    sk = Sketch()
    p1 = add_point!(sk, -0.1, 0.1)
    p2 = add_point!(sk, 4.1, -0.1)
    p3 = add_point!(sk, 4.2, 3.0)
    p4 = add_point!(sk, -0.1, 3.1)

    l1 = push!(sk, Line(p1, p2))
    l2 = push!(sk, Line(p2, p3))
    l3 = push!(sk, Line(p3, p4))
    l4 = push!(sk, Line(p4, p1))

    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, FixedPoint(p2, 4.0, 0.0))
    push!(sk, FixedPoint(p4, 0.0, 3.0))

    push!(sk, Horizontal(l1))
    push!(sk, Vertical(l2))
    push!(sk, Horizontal(l3))
    push!(sk, Vertical(l4))

    center = add_point!(sk, 1.1, 1.0)
    rim = add_point!(sk, 1.1, 3.0)
    p5 = add_point!(sk, 2.8, 1.2)
    c1 = push!(sk, Circle(center, rim))
    l5 = push!(sk, Line(center, rim))

    push!(sk, FixedPoint(center, 1.0, 1.0))
    push!(sk, Diameter(c1, 4.0))
    push!(sk, Vertical(l5))
    push!(sk, CircleCoincident(c1, p5))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test stats.status == :converged

    max_iters = 0
    for k in 1:10
        dx = 0.05 * sin(0.2 * k)
        dy = 0.05 * cos(0.2 * k)
        set_point!(sk, p1, dx, dy)
        stats = solve!(sk; options=LOG_OPTIONS)
        max_iters = max(max_iters, stats.iters)
        @test stats.status == :converged
        ix1, iy1 = 2 * (p1 - 1) + 1, 2 * (p1 - 1) + 2
        @test isapprox(sk.x[ix1], 0.0; atol=1e-6)
        @test isapprox(sk.x[iy1], 0.0; atol=1e-6)
    end
    @test max_iters >= 1
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
    # Rectangle with duplicated corner points plus a circle point tied to the top edge.
    sk = Sketch()
    p1a = add_point!(sk, 1.6, -0.9)
    p1b = add_point!(sk, 0.6, -0.4)
    p2a = add_point!(sk, 4.8, 0.9)
    p2b = add_point!(sk, 4.2, -0.6)
    p3a = add_point!(sk, 5.2, 3.8)
    p3b = add_point!(sk, 4.6, 3.2)
    p4a = add_point!(sk, 0.5, 3.9)
    p4b = add_point!(sk, 1.2, 3.1)

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

    # Circle point constrained to the top edge.
    p5 = add_point!(sk, 2.2, 2.8)    # center
    p6 = add_point!(sk, 2.4, -0.2)   # rim
    p7 = add_point!(sk, 1.3, 2.4)    # point on circle
    p8 = add_point!(sk, 1.0, -1.0)   # anchor for vertical line

    c1 = push!(sk, Circle(p5, p6))
    l5 = push!(sk, Line(p7, p8))
    l6 = push!(sk, Line(p7, p3a))

    push!(sk, FixedPoint(p5, 1.0, 1.0))
    push!(sk, FixedPoint(p8, 1.0, -1.0))
    push!(sk, Diameter(c1, 4.0))
    push!(sk, Vertical(l5))
    push!(sk, Horizontal(l6))
    push!(sk, CircleCoincident(c1, p7))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test stats.status == :converged

    ix3a, iy3a = 2 * (p3a - 1) + 1, 2 * (p3a - 1) + 2
    @test isapprox(sk.x[ix3a], 4.0; atol=1e-6)
    @test isapprox(sk.x[iy3a], 3.0; atol=1e-6)
    ix7, iy7 = 2 * (p7 - 1) + 1, 2 * (p7 - 1) + 2
    @test isapprox(sk.x[ix7], 1.0; atol=1e-6)
    @test isapprox(sk.x[iy7], 3.0; atol=1e-6)

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

    stats = solve!(sk; options=LOG_OPTIONS)
    ixr, iyr = 2 * (rim - 1) + 1, 2 * (rim - 1) + 2
    @test stats.status == :converged
    @test isapprox(sk.x[ixr], 5.0; atol=1e-6)
    @test isapprox(sk.x[iyr], 0.0; atol=1e-6)
end

@testset "conflicting constraints" begin
    # A single point fixed at two different x-positions is inconsistent (residual stays nonzero).
    sk = Sketch()
    p1 = add_point!(sk, 0.0, 0.0)

    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, FixedPoint(p1, 1.0, 0.0))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test residual_norm(stats) > 1e-2
    @test has_conflict(stats; tol=1e-3)
end

@testset "conflicting line axes" begin
    # Fixed endpoints at (0,0) and (1,1) cannot satisfy both horizontal and vertical constraints.
    sk = Sketch()
    p1 = add_point!(sk, 0.0, 0.0)
    p2 = add_point!(sk, 1.0, 1.0)
    l1 = push!(sk, Line(p1, p2))

    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, FixedPoint(p2, 1.0, 1.0))
    push!(sk, Horizontal(l1))
    push!(sk, Vertical(l1))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test residual_norm(stats) > 1e-2
    @test has_conflict(stats; tol=1e-3)
end

@testset "conflicting parallel lines" begin
    # Two perpendicular fixed lines cannot be made parallel.
    sk = Sketch()
    p1 = add_point!(sk, 0.0, 0.0)
    p2 = add_point!(sk, 1.0, 0.0)
    p3 = add_point!(sk, 0.0, 0.0)
    p4 = add_point!(sk, 0.0, 1.0)
    l1 = push!(sk, Line(p1, p2))
    l2 = push!(sk, Line(p3, p4))

    push!(sk, FixedPoint(p1, 0.0, 0.0))
    push!(sk, FixedPoint(p2, 1.0, 0.0))
    push!(sk, FixedPoint(p3, 0.0, 0.0))
    push!(sk, FixedPoint(p4, 0.0, 1.0))
    push!(sk, Parallel(l1, l2))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test residual_norm(stats) > 1e-2
    @test has_conflict(stats; tol=1e-3)
end

@testset "conflicting diameter constraint" begin
    # Fixed circle center/rim do not match the requested diameter.
    sk = Sketch()
    center = add_point!(sk, 0.0, 0.0)
    rim = add_point!(sk, 1.0, 0.0)
    c1 = push!(sk, Circle(center, rim))

    push!(sk, FixedPoint(center, 0.0, 0.0))
    push!(sk, FixedPoint(rim, 1.0, 0.0))
    push!(sk, Diameter(c1, 10.0))

    stats = solve!(sk; options=LOG_OPTIONS)
    @test residual_norm(stats) > 1.0
    @test has_conflict(stats; tol=1e-3)
end
