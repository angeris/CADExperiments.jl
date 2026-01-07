using CADConstraints

# Allocation smoke test: point updates should allocate 0 bytes; solve! is reported.
function run_allocations()
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

    solve!(sk) # warm up
    GC.gc()
    alloc_move = @allocated set_point!(sk, p2, 3.0, -1.0)
    GC.gc()
    alloc_solve = @allocated solve!(sk)
    return alloc_move, alloc_solve
end

alloc_move, alloc_solve = run_allocations()
println("set_point! allocations (bytes): ", alloc_move)
println("solve! allocations (bytes): ", alloc_solve)
