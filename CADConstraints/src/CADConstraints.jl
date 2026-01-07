module CADConstraints

using SparseArrays
using SparseLNNS

import Base: push!

export Shape, Constraint, Line, Circle, Arc
export FixedPoint, Coincident, CircleCoincident, Horizontal, Vertical, Parallel, Distance, Diameter, Normal
export Sketch, add_point!, set_point!, build_problem!, solve!
export residual_norm, has_conflict, conflicts, ConflictEntry, ConflictReport

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
    Circle(center, rim)

Circle shape defined by center and rim point indices.
"""
struct Circle <: Shape
    center::Int
    rim::Int
end

"""
    Arc(center, start, finish)

Arc shape defined by center, start, and end point indices.
"""
struct Arc <: Shape
    center::Int
    start::Int
    finish::Int
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
    Distance(p1, p2, d)

Constrain the distance between points `p1` and `p2` to `d`.
"""
struct Distance{T<:Real} <: Constraint
    p1::Int
    p2::Int
    d::T
end

"""
    Diameter(circle, d)

Constrain a circle's diameter to `d`.
"""
struct Diameter{T<:Real} <: Constraint
    circle::Int
    d::T
end

"""
    Normal(circle, line)

Constrain a line to pass through a circle center (normal direction).
"""
struct Normal <: Constraint
    circle::Int
    line::Int
end

"""
    CircleCoincident(circle, p)

Constrain point `p` to lie on the circle.
"""
struct CircleCoincident <: Constraint
    circle::Int
    p::Int
end

"""
    Sketch()

Container for points, shapes, constraints, and cached solver state.
"""
mutable struct Sketch
    x::Vector{Float64}
    circles::Vector{Circle}
    arcs::Vector{Arc}
    lines::Vector{Line}
    constraints::Vector{Constraint}
    structure_dirty::Bool
    value_dirty::Bool
    problem::Problem
    state::State
    work::Workspace
end

"""
    ConflictEntry

Per-constraint conflict entry with index, kind, and residual norm.
"""
struct ConflictEntry
    index::Int
    kind::Symbol
    norm::Float64
end

"""
    ConflictReport

Summary of constraint consistency for the latest solve.
"""
struct ConflictReport
    residual_norm::Float64
    conflicted::Bool
    entries::Vector{ConflictEntry}
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
    return Sketch(Float64[], Circle[], Arc[], Line[], Constraint[], true, false, problem, state, work)
end

@inline function point_indices(p)
    ix = 2 * (p - 1) + 1
    return ix, ix + 1
end

@inline function line_points(sketch, line_idx)
    line = sketch.lines[line_idx]
    return line.p1, line.p2
end

@inline function circle_points(sketch, circle_idx)
    circle = sketch.circles[circle_idx]
    return circle.center, circle.rim
end

@inline function mark_structure_dirty!(sketch::Sketch)
    sketch.structure_dirty = true
    sketch.value_dirty = true
    return nothing
end

@inline function mark_value_dirty!(sketch::Sketch)
    sketch.value_dirty = true
    return nothing
end

"""
    add_point!(sketch, x, y) -> point_index

Append a point `(x, y)` and return its 1-based index.
"""
function add_point!(sketch::Sketch, x, y)
    push!(sketch.x, Float64(x))
    push!(sketch.x, Float64(y))
    mark_structure_dirty!(sketch)
    return length(sketch.x) ÷ 2
end

"""
    set_point!(sketch, p, x, y)

Update point `p` to `(x, y)` without changing structure.
"""
function set_point!(sketch::Sketch, p, x, y)
    ix, iy = point_indices(p)
    sketch.x[ix] = Float64(x)
    sketch.x[iy] = Float64(y)
    mark_value_dirty!(sketch)
    return nothing
end

function push!(sketch::Sketch, line::Line)
    push!(sketch.lines, line)
    mark_structure_dirty!(sketch)
    return length(sketch.lines)
end

function push!(sketch::Sketch, circle::Circle)
    push!(sketch.circles, circle)
    mark_structure_dirty!(sketch)
    return length(sketch.circles)
end

function push!(sketch::Sketch, arc::Arc)
    push!(sketch.arcs, arc)
    mark_structure_dirty!(sketch)
    return length(sketch.arcs)
end

constraint_rows(::FixedPoint) = 2
constraint_rows(::Coincident) = 2
constraint_rows(::Horizontal) = 1
constraint_rows(::Vertical) = 1
constraint_rows(::Parallel) = 1
constraint_rows(::Distance) = 1
constraint_rows(::Diameter) = 1
constraint_rows(::Normal) = 1
constraint_rows(::CircleCoincident) = 1

function push!(sketch::Sketch, constraint::FixedPoint)
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Coincident)
    if constraint.p1 == constraint.p2
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Horizontal)
    p1, p2 = line_points(sketch, constraint.line)
    if p1 == p2
        push!(sketch, Coincident(p1, p2))
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Vertical)
    p1, p2 = line_points(sketch, constraint.line)
    if p1 == p2
        push!(sketch, Coincident(p1, p2))
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
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
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Distance)
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Diameter)
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::Normal)
    p1, p2 = line_points(sketch, constraint.line)
    if p1 == p2
        push!(sketch, Coincident(p1, p2))
        return constraint
    end
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
    return constraint
end

function push!(sketch::Sketch, constraint::CircleCoincident)
    push!(sketch.constraints, constraint)
    mark_structure_dirty!(sketch)
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

function pattern!(Jpat, constraint::Distance, sketch, offset)
    ix1, iy1 = point_indices(constraint.p1)
    ix2, iy2 = point_indices(constraint.p2)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, iy1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    Jpat[offset + 1, iy2] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::Diameter, sketch, offset)
    center, rim = circle_points(sketch, constraint.circle)
    ix1, iy1 = point_indices(center)
    ix2, iy2 = point_indices(rim)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, iy1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    Jpat[offset + 1, iy2] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::Normal, sketch, offset)
    center, _ = circle_points(sketch, constraint.circle)
    p1, p2 = line_points(sketch, constraint.line)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    icx, icy = point_indices(center)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, iy1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    Jpat[offset + 1, iy2] = 1.0
    Jpat[offset + 1, icx] = 1.0
    Jpat[offset + 1, icy] = 1.0
    return nothing
end

function pattern!(Jpat, constraint::CircleCoincident, sketch, offset)
    center, rim = circle_points(sketch, constraint.circle)
    ix1, iy1 = point_indices(center)
    ix2, iy2 = point_indices(rim)
    ixp, iyp = point_indices(constraint.p)
    Jpat[offset + 1, ix1] = 1.0
    Jpat[offset + 1, iy1] = 1.0
    Jpat[offset + 1, ix2] = 1.0
    Jpat[offset + 1, iy2] = 1.0
    Jpat[offset + 1, ixp] = 1.0
    Jpat[offset + 1, iyp] = 1.0
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

function residual!(out, x, constraint::Distance, sketch, offset)
    ix1, iy1 = point_indices(constraint.p1)
    ix2, iy2 = point_indices(constraint.p2)
    dx = x[ix2] - x[ix1]
    dy = x[iy2] - x[iy1]
    out[offset + 1] = dx * dx + dy * dy - constraint.d * constraint.d
    return nothing
end

function residual!(out, x, constraint::Diameter, sketch, offset)
    center, rim = circle_points(sketch, constraint.circle)
    ix1, iy1 = point_indices(center)
    ix2, iy2 = point_indices(rim)
    dx = x[ix2] - x[ix1]
    dy = x[iy2] - x[iy1]
    r2 = 0.25 * constraint.d * constraint.d
    out[offset + 1] = dx * dx + dy * dy - r2
    return nothing
end

function residual!(out, x, constraint::Normal, sketch, offset)
    center, _ = circle_points(sketch, constraint.circle)
    p1, p2 = line_points(sketch, constraint.line)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    icx, icy = point_indices(center)
    dx = x[ix2] - x[ix1]
    dy = x[iy2] - x[iy1]
    dx1 = x[icx] - x[ix1]
    dy1 = x[icy] - x[iy1]
    out[offset + 1] = dx * dy1 - dy * dx1
    return nothing
end

function residual!(out, x, constraint::CircleCoincident, sketch, offset)
    center, rim = circle_points(sketch, constraint.circle)
    ix1, iy1 = point_indices(center)
    ix2, iy2 = point_indices(rim)
    ixp, iyp = point_indices(constraint.p)
    dxp = x[ixp] - x[ix1]
    dyp = x[iyp] - x[iy1]
    dxr = x[ix2] - x[ix1]
    dyr = x[iy2] - x[iy1]
    out[offset + 1] = dxp * dxp + dyp * dyp - (dxr * dxr + dyr * dyr)
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

function jacobian!(J, x, constraint::Distance, sketch, offset)
    ix1, iy1 = point_indices(constraint.p1)
    ix2, iy2 = point_indices(constraint.p2)
    dx = x[ix2] - x[ix1]
    dy = x[iy2] - x[iy1]
    J[offset + 1, ix1] = -2.0 * dx
    J[offset + 1, iy1] = -2.0 * dy
    J[offset + 1, ix2] = 2.0 * dx
    J[offset + 1, iy2] = 2.0 * dy
    return nothing
end

function jacobian!(J, x, constraint::Diameter, sketch, offset)
    center, rim = circle_points(sketch, constraint.circle)
    ix1, iy1 = point_indices(center)
    ix2, iy2 = point_indices(rim)
    dx = x[ix2] - x[ix1]
    dy = x[iy2] - x[iy1]
    J[offset + 1, ix1] = -2.0 * dx
    J[offset + 1, iy1] = -2.0 * dy
    J[offset + 1, ix2] = 2.0 * dx
    J[offset + 1, iy2] = 2.0 * dy
    return nothing
end

function jacobian!(J, x, constraint::Normal, sketch, offset)
    center, _ = circle_points(sketch, constraint.circle)
    p1, p2 = line_points(sketch, constraint.line)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    icx, icy = point_indices(center)
    x1 = x[ix1]
    y1 = x[iy1]
    x2 = x[ix2]
    y2 = x[iy2]
    cx = x[icx]
    cy = x[icy]
    dx = x2 - x1
    dy = y2 - y1
    J[offset + 1, ix1] = y2 - cy
    J[offset + 1, iy1] = cx - x2
    J[offset + 1, ix2] = cy - y1
    J[offset + 1, iy2] = x1 - cx
    J[offset + 1, icx] = -dy
    J[offset + 1, icy] = dx
    return nothing
end

function jacobian!(J, x, constraint::CircleCoincident, sketch, offset)
    center, rim = circle_points(sketch, constraint.circle)
    ix1, iy1 = point_indices(center)
    ix2, iy2 = point_indices(rim)
    ixp, iyp = point_indices(constraint.p)
    dxp = x[ixp] - x[ix1]
    dyp = x[iyp] - x[iy1]
    dxr = x[ix2] - x[ix1]
    dyr = x[iy2] - x[iy1]
    J[offset + 1, ix1] = 2.0 * (dxr - dxp)
    J[offset + 1, iy1] = 2.0 * (dyr - dyp)
    J[offset + 1, ix2] = -2.0 * dxr
    J[offset + 1, iy2] = -2.0 * dyr
    J[offset + 1, ixp] = 2.0 * dxp
    J[offset + 1, iyp] = 2.0 * dyp
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
        sketch.value_dirty = false
    elseif sketch.value_dirty
        copyto!(sketch.state.x, sketch.x)
        sketch.value_dirty = false
    end
    stats = SparseLNNS.solve!(sketch.state, sketch.problem, sketch.work; options=options)
    copyto!(sketch.x, sketch.state.x)
    return stats
end

"""
    residual_norm(stats)

Return the residual norm `sqrt(2 * cost)` from the last solve.
"""
function residual_norm(stats)
    return sqrt(2 * stats.cost)
end

"""
    has_conflict(stats; tol=1e-6)

Return true if the residual norm exceeds `tol`, indicating inconsistent constraints.
"""
function has_conflict(stats; tol=1e-6)
    return residual_norm(stats) > tol
end

"""
    conflicts(sketch; tol=1e-6, max_items=5)
    conflicts(sketch, stats; tol=1e-6, max_items=5)

Return a conflict report based on the current residuals. Call after `solve!`.
"""
function conflicts(sketch::Sketch; tol=1e-6, max_items=5)
    return conflicts(sketch, sketch.state.stats; tol=tol, max_items=max_items)
end

function conflicts(sketch::Sketch, stats; tol=1e-6, max_items=5)
    if sketch.structure_dirty
        throw(ArgumentError("conflicts requires an up-to-date solve; call solve! first"))
    end
    if sketch.value_dirty
        sketch.problem.r!(sketch.work.r, sketch.x)
    end

    resnorm = residual_norm(stats)
    conflicted = resnorm > tol
    entries = ConflictEntry[]
    if !conflicted || max_items <= 0
        return ConflictReport(resnorm, conflicted, entries)
    end

    offset = 0
    for (idx, constraint) in enumerate(sketch.constraints)
        rows = constraint_rows(constraint)
        sumsq = 0.0
        @inbounds for j in 1:rows
            r = sketch.work.r[offset + j]
            sumsq += r * r
        end
        offset += rows
        norm = sqrt(sumsq)
        if norm <= tol
            continue
        end

        entry = ConflictEntry(idx, nameof(typeof(constraint)), norm)
        if length(entries) < max_items
            push!(entries, entry)
        elseif norm > entries[end].norm
            entries[end] = entry
        else
            continue
        end

        i = length(entries)
        while i > 1 && entries[i].norm > entries[i - 1].norm
            entries[i], entries[i - 1] = entries[i - 1], entries[i]
            i -= 1
        end
    end

    return ConflictReport(resnorm, conflicted, entries)
end

end # module CADConstraints
