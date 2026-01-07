using CADConstraints
using SparseLNNS: Options

function build_wiggle_sketch()
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

    # Circle tied to the rectangle: center at (1,1), rim fixed by diameter + distance to top-right.
    p5 = add_point!(sk, 1.2, 1.1)   # center
    p6 = add_point!(sk, 1.1, 3.2)   # rim
    c1 = push!(sk, Circle(p5, p6))
    l5 = push!(sk, Line(p5, p6))

    push!(sk, FixedPoint(p5, 1.0, 1.0))
    push!(sk, Diameter(c1, 4.0))
    push!(sk, Vertical(l5))
    push!(sk, Distance(p6, p3a, 3.0))

    # Distance-only anchor: p7 should land at (2, 0) between fixed endpoints.
    p7 = add_point!(sk, 1.7, 0.2)
    push!(sk, Distance(p7, p1a, 2.0))
    push!(sk, Distance(p7, p2a, 2.0))

    return sk, p1a, 0.0, 0.0
end

function wiggle!(sketch, anchor, base_x, base_y, steps, options; print_stats=false)
    total_iters = 0
    max_iters = 0
    set_point!(sketch, anchor, base_x, base_y)
    for k in 1:steps
        dx = 0.05 * sin(0.2 * k)
        dy = 0.05 * cos(0.2 * k)
        set_point!(sketch, anchor, base_x + dx, base_y + dy)
        stats = solve!(sketch; options=options)
        if print_stats
            println("step ", k, " iters=", stats.iters, " status=", stats.status, " cost=", stats.cost)
        end
        total_iters += stats.iters
        if stats.iters > max_iters
            max_iters = stats.iters
        end
    end
    return total_iters, max_iters
end

sk, anchor, base_x, base_y = build_wiggle_sketch()
options = Options()
solve!(sk; options=options) # warm up

steps = 100
alloc = @allocated wiggle!(sk, anchor, base_x, base_y, steps, options; print_stats=true)
t = @elapsed begin
    total_iters, max_iters = wiggle!(sk, anchor, base_x, base_y, steps, options)
    println("wiggle steps: ", steps)
    println("total iters: ", total_iters, " max iters: ", max_iters)
end

println("wiggle time (sec): ", t, " avg (ms/solve): ", 1000 * t / steps)
println("wiggle alloc (bytes): ", alloc, " per solve: ", alloc / steps)
