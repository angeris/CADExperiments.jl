using CADConstraints

# Timing smoke test for a moderately constrained rectangle with coincident endpoints.
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

solve!(sk) # warm up

t = @elapsed stats = solve!(sk)
println("solve time (sec): ", t, " iters=", stats.iters, " status=", stats.status)
