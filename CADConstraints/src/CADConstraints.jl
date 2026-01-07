module CADConstraints

using SparseArrays
using SparseLNNS

import Base: push!

export Shape, Constraint, Line
export FixedPoint, Coincident, Horizontal, Vertical, Parallel
export Sketch, add_point!, build_problem!, solve!

abstract type Shape end
abstract type Constraint end

"""
    Line(p1, p2)

Line shape defined by two point indices.
"""
struct Line <: Shape
    p1::Int
    p2::Int
end

"""
    FixedPoint(p, x, y)

Fix point `p` at `(x, y)`.
"""
struct FixedPoint{T<:Real} <: Constraint
    p::Int
    x::T
    y::T
end

"""
    Coincident(p1, p2)

Constrain points `p1` and `p2` to coincide.
"""
struct Coincident <: Constraint
    p1::Int
    p2::Int
end

"""
    Horizontal(line)

Constrain a line to be horizontal (y1 == y2).
"""
struct Horizontal <: Constraint
    line::Int
end

"""
    Vertical(line)

Constrain a line to be vertical (x1 == x2).
"""
struct Vertical <: Constraint
    line::Int
end

"""
    Parallel(line1, line2)

Constrain two lines to be parallel.
"""
struct Parallel <: Constraint
    line1::Int
    line2::Int
end

"""
    Sketch()

Container for points, shapes, constraints, and cached solver state.
"""
mutable struct Sketch
    x::Vector{Float64}
    lines::Vector{Line}
    constraints::Vector{Constraint}
    structure_dirty::Bool
    problem::Problem
    state::State
    work::Workspace
end

function empty_state_work()
    Jpat = spzeros(Float64, 0, 0)
    r! = function (out, x)
        return nothing
    end
    J! = function (J, x)
        return nothing
    end
    problem = Problem(r!, J!, Jpat)
    state, work = initialize(problem, Float64[])
    return problem, state, work
end

function Sketch()
    problem, state, work = empty_state_work()
    return Sketch(Float64[], Line[], Constraint[], true, problem, state, work)
end

@inline function point_indices(p)
    ix = 2 * (p - 1) + 1
    return ix, ix + 1
end

@inline function line_points(sketch, line_idx)
    line = sketch.lines[line_idx]
    return line.p1, line.p2
end

@inline function mark_dirty!(sketch::Sketch)
    sketch.structure_dirty = true
    return nothing
end

"""
    add_point!(sketch, x, y) -> point_index

Append a point `(x, y)` and return its 1-based index.
"""
function add_point!(sketch::Sketch, x, y)
    push!(sketch.x, Float64(x))
    push!(sketch.x, Float64(y))
    mark_dirty!(sketch)
    return length(sketch.x) ÷ 2
end

function push!(sketch::Sketch, line::Line)
    push!(sketch.lines, line)
    mark_dirty!(sketch)
    return length(sketch.lines)
end

constraint_rows(::FixedPoint) = 2
constraint_rows(::Coincident) = 2
constraint_rows(::Horizontal) = 1
constraint_rows(::Vertical) = 1
constraint_rows(::Parallel) = 1

function push!(sketch::Sketch, constraint::FixedPoint)
    push!(sketch.constraints, constraint)
    mark_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Coincident)
    if constraint.p1 == constraint.p2
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Horizontal)
    p1, p2 = line_points(sketch, constraint.line)
    if p1 == p2
        push!(sketch, Coincident(p1, p2))
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Vertical)
    p1, p2 = line_points(sketch, constraint.line)
    if p1 == p2
        push!(sketch, Coincident(p1, p2))
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Parallel)
    p1, p2 = line_points(sketch, constraint.line1)
    p3, p4 = line_points(sketch, constraint.line2)
    if p1 == p2 || p3 == p4
        if p1 == p2
            push!(sketch, Coincident(p1, p2))
        end
        if p3 == p4
            push!(sketch, Coincident(p3, p4))
        end
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_dirty!(sketch)
    return constraint
end

function pattern!(Jpat, constraint::FixedPoint, sketch, offset)
    ix, iy = point_indices(constraint.p)
    Jpat[offset + 1, ix] = 1.0
    Jpat[offset + 2, iy] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::Coincident, sketch, offset)
    ix1, iy1 = point_indices(constraint.p1)
    ix2, iy2 = point_indices(constraint.p2)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    Jpat[offset + 2, iy1] = 1.0
    Jpat[offset + 2, iy2] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::Horizontal, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line)
    _, iy1 = point_indices(p1)
    _, iy2 = point_indices(p2)
    Jpat[offset + 1, iy1] = 1.0
    Jpat[offset + 1, iy2] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::Vertical, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line)
    ix1, _ = point_indices(p1)
    ix2, _ = point_indices(p2)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::Parallel, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line1)
    p3, p4 = line_points(sketch, constraint.line2)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    ix3, iy3 = point_indices(p3)
    ix4, iy4 = point_indices(p4)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, iy1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    Jpat[offset + 1, iy2] = 1.0
    Jpat[offset + 1, ix3] = 1.0
    Jpat[offset + 1, iy3] = 1.0
    Jpat[offset + 1, ix4] = 1.0
    Jpat[offset + 1, iy4] = 1.0
    return nothing
end

function residual!(out, x, constraint::FixedPoint, sketch, offset)
    ix, iy = point_indices(constraint.p)
    out[offset + 1] = x[ix] - constraint.x
    out[offset + 2] = x[iy] - constraint.y
    return nothing
end

function residual!(out, x, constraint::Coincident, sketch, offset)
    ix1, iy1 = point_indices(constraint.p1)
    ix2, iy2 = point_indices(constraint.p2)
    out[offset + 1] = x[ix1] - x[ix2]
    out[offset + 2] = x[iy1] - x[iy2]
    return nothing
end

function residual!(out, x, constraint::Horizontal, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line)
    _, iy1 = point_indices(p1)
    _, iy2 = point_indices(p2)
    out[offset + 1] = x[iy1] - x[iy2]
    return nothing
end

function residual!(out, x, constraint::Vertical, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line)
    ix1, _ = point_indices(p1)
    ix2, _ = point_indices(p2)
    out[offset + 1] = x[ix1] - x[ix2]
    return nothing
end

function residual!(out, x, constraint::Parallel, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line1)
    p3, p4 = line_points(sketch, constraint.line2)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    ix3, iy3 = point_indices(p3)
    ix4, iy4 = point_indices(p4)
    dx12 = x[ix2] - x[ix1]
    dy12 = x[iy2] - x[iy1]
    dx34 = x[ix4] - x[ix3]
    dy34 = x[iy4] - x[iy3]
    out[offset + 1] = dx12 * dy34 - dy12 * dx34
    return nothing
end

function jacobian!(J, x, constraint::FixedPoint, sketch, offset)
    ix, iy = point_indices(constraint.p)
    J[offset + 1, ix] = 1.0
    J[offset + 2, iy] = 1.0
    return nothing
end

function jacobian!(J, x, constraint::Coincident, sketch, offset)
    ix1, iy1 = point_indices(constraint.p1)
    ix2, iy2 = point_indices(constraint.p2)
    J[offset + 1, ix1] = 1.0
    J[offset + 1, ix2] = -1.0
    J[offset + 2, iy1] = 1.0
    J[offset + 2, iy2] = -1.0
    return nothing
end

function jacobian!(J, x, constraint::Horizontal, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line)
    _, iy1 = point_indices(p1)
    _, iy2 = point_indices(p2)
    J[offset + 1, iy1] = 1.0
    J[offset + 1, iy2] = -1.0
    return nothing
end

function jacobian!(J, x, constraint::Vertical, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line)
    ix1, _ = point_indices(p1)
    ix2, _ = point_indices(p2)
    J[offset + 1, ix1] = 1.0
    J[offset + 1, ix2] = -1.0
    return nothing
end

function jacobian!(J, x, constraint::Parallel, sketch, offset)
    p1, p2 = line_points(sketch, constraint.line1)
    p3, p4 = line_points(sketch, constraint.line2)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    ix3, iy3 = point_indices(p3)
    ix4, iy4 = point_indices(p4)
    dx12 = x[ix2] - x[ix1]
    dy12 = x[iy2] - x[iy1]
    dx34 = x[ix4] - x[ix3]
    dy34 = x[iy4] - x[iy3]
    J[offset + 1, ix1] = -dy34
    J[offset + 1, iy1] = dx34
    J[offset + 1, ix2] = dy34
    J[offset + 1, iy2] = -dx34
    J[offset + 1, ix3] = dy12
    J[offset + 1, iy3] = -dx12
    J[offset + 1, ix4] = -dy12
    J[offset + 1, iy4] = dx12
    return nothing
end

"""
    build_problem!(sketch)

Construct a SparseLNNS problem from current constraints.
"""
function build_problem!(sketch::Sketch)
    m = 0
    for constraint in sketch.constraints
        m += constraint_rows(constraint)
    end
    n = length(sketch.x)
    m > 0 || throw(ArgumentError("cannot build problem with no constraints"))
    n > 0 || throw(ArgumentError("cannot build problem with no points"))

    Jpat = spzeros(Float64, m, n)
    offset = 0
    for constraint in sketch.constraints
        pattern!(Jpat, constraint, sketch, offset)
        offset += constraint_rows(constraint)
    end

    r! = function (out, x)
        fill!(out, zero(eltype(out)))
        offset = 0
        for constraint in sketch.constraints
            residual!(out, x, constraint, sketch, offset)
            offset += constraint_rows(constraint)
        end
        return nothing
    end

    J! = function (J, x)
        fill!(nonzeros(J), zero(eltype(nonzeros(J))))
        offset = 0
        for constraint in sketch.constraints
            jacobian!(J, x, constraint, sketch, offset)
            offset += constraint_rows(constraint)
        end
        return nothing
    end

    sketch.problem = Problem(r!, J!, Jpat)
    sketch.structure_dirty = false
    return sketch.problem
end

"""
    solve!(sketch; options=SparseLNNS.Options())

Solve the current constraint system and update `sketch.x` in place.
"""
function solve!(sketch::Sketch; options=Options())
    if sketch.structure_dirty
        build_problem!(sketch)
        sketch.state, sketch.work = initialize(sketch.problem, sketch.x; options=options)
    else
        copyto!(sketch.state.x, sketch.x)
    end
    stats = SparseLNNS.solve!(sketch.state, sketch.problem, sketch.work; options=options)
    copyto!(sketch.x, sketch.state.x)
    return stats
end

end # module CADConstraints
