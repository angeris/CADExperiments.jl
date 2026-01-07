using SparseArrays
using NLLS

r!(out, x) = (out[1] = x[1] - 1.0)
function J!(J, x)
    nonzeros(J)[1] = 1.0
    return nothing
end

Jpat = sparse([1], [1], [1.0], 1, 1)
prob = Problem(r!, J!, Jpat)
state, work = initialize(prob, [0.0])

solve!(state, prob, work) # warm up
alloc = @allocated solve!(state, prob, work)
println("solve! allocations (bytes): ", alloc)
