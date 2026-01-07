module CADConstraints

using SparseArrays
using SparseLNNS

export Sketch, Line, Constraint
export add_point!, add_line!, add_fixed_point!, add_coincident!
export add_horizontal!, add_vertical!, add_parallel!
export build_problem!, solve!

"""
    Line(p1, p2)

Line shape defined by two point indices.
"""
struct Line
    p1::Int
    p2::Int
end

"""
    Constraint(m, rows, cols, residual!, jacobian!)

Point-based constraint with a fixed Jacobian pattern.
"""
struct Constraint
    m::Int
    rows::Vector{Int}
    cols::Vector{Int}
    residual!::Function
    jacobian!::Function
end

"""
    Sketch()

Container for points, shapes, constraints, and cached solver state.
"""
mutable struct Sketch{T<:AbstractFloat}
    x::Vector{T}
    lines::Vector{Line}
    constraints::Vector{Constraint}
    m::Int
    structure_dirty::Bool
    problem::Union{Nothing, SparseLNNS.Problem{T}}
    state::Union{Nothing, SparseLNNS.State{T}}
    work::Union{Nothing, SparseLNNS.Workspace{T}}
end

Sketch{T}() where {T<:AbstractFloat} = Sketch{T}(T[], Line[], Constraint[], 0, true, nothing, nothing, nothing)
Sketch() = Sketch{Float64}()

@inline function point_indices(p::Int)
    ix = 2 * (p - 1) + 1
    return ix, ix + 1
end

@inline function line_points(sketch::Sketch, line_idx::Int)
    line = sketch.lines[line_idx]
    return line.p1, line.p2
end

@inline function mark_dirty!(sketch::Sketch)
    sketch.structure_dirty = true
    sketch.problem = nothing
    sketch.state = nothing
    sketch.work = nothing
    return nothing
end

"""
    add_point!(sketch, x, y) -> point_index

Append a point `(x, y)` and return its 1-based index.
"""
function add_point!(sketch::Sketch{T}, x::Real, y::Real) where {T<:AbstractFloat}
    push!(sketch.x, T(x))
    push!(sketch.x, T(y))
    mark_dirty!(sketch)
    return length(sketch.x) ÷ 2
end

"""
    add_line!(sketch, p1, p2) -> line_index

Create a line shape from two point indices.
"""
function add_line!(sketch::Sketch, p1::Int, p2::Int)
    push!(sketch.lines, Line(p1, p2))
    mark_dirty!(sketch)
    return length(sketch.lines)
end

function add_constraint!(sketch::Sketch, constraint::Constraint)
    push!(sketch.constraints, constraint)
    sketch.m += constraint.m
    mark_dirty!(sketch)
    return constraint
end

"""
    add_fixed_point!(sketch, p, x0, y0)

Fix point `p` at `(x0, y0)`.
"""
function add_fixed_point!(sketch::Sketch, p::Int, x0::Real, y0::Real)
    ix, iy = point_indices(p)
    rows = [1, 2]
    cols = [ix, iy]
    residual! = function (out, x, offset)
        out[offset + 1] = x[ix] - x0
        out[offset + 2] = x[iy] - y0
        return nothing
    end
    jacobian! = function (J, x, offset)
        J[offset + 1, ix] = 1.0
        J[offset + 2, iy] = 1.0
        return nothing
    end
    return add_constraint!(sketch, Constraint(2, rows, cols, residual!, jacobian!))
end

"""
    add_coincident!(sketch, p1, p2)

Constrain points `p1` and `p2` to coincide.
"""
function add_coincident!(sketch::Sketch, p1::Int, p2::Int)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    rows = [1, 1, 2, 2]
    cols = [ix1, ix2, iy1, iy2]
    residual! = function (out, x, offset)
        out[offset + 1] = x[ix1] - x[ix2]
        out[offset + 2] = x[iy1] - x[iy2]
        return nothing
    end
    jacobian! = function (J, x, offset)
        J[offset + 1, ix1] = 1.0
        J[offset + 1, ix2] = -1.0
        J[offset + 2, iy1] = 1.0
        J[offset + 2, iy2] = -1.0
        return nothing
    end
    return add_constraint!(sketch, Constraint(2, rows, cols, residual!, jacobian!))
end

"""
    add_horizontal!(sketch, line_idx)

Constrain a line to be horizontal (y1 == y2).
"""
function add_horizontal!(sketch::Sketch, line_idx::Int)
    p1, p2 = line_points(sketch, line_idx)
    _, iy1 = point_indices(p1)
    _, iy2 = point_indices(p2)
    rows = [1, 1]
    cols = [iy1, iy2]
    residual! = function (out, x, offset)
        out[offset + 1] = x[iy1] - x[iy2]
        return nothing
    end
    jacobian! = function (J, x, offset)
        J[offset + 1, iy1] = 1.0
        J[offset + 1, iy2] = -1.0
        return nothing
    end
    return add_constraint!(sketch, Constraint(1, rows, cols, residual!, jacobian!))
end

"""
    add_vertical!(sketch, line_idx)

Constrain a line to be vertical (x1 == x2).
"""
function add_vertical!(sketch::Sketch, line_idx::Int)
    p1, p2 = line_points(sketch, line_idx)
    ix1, _ = point_indices(p1)
    ix2, _ = point_indices(p2)
    rows = [1, 1]
    cols = [ix1, ix2]
    residual! = function (out, x, offset)
        out[offset + 1] = x[ix1] - x[ix2]
        return nothing
    end
    jacobian! = function (J, x, offset)
        J[offset + 1, ix1] = 1.0
        J[offset + 1, ix2] = -1.0
        return nothing
    end
    return add_constraint!(sketch, Constraint(1, rows, cols, residual!, jacobian!))
end

"""
    add_parallel!(sketch, line1_idx, line2_idx)

Constrain two lines to be parallel using a cross-product residual.
"""
function add_parallel!(sketch::Sketch, line1_idx::Int, line2_idx::Int)
    p1, p2 = line_points(sketch, line1_idx)
    p3, p4 = line_points(sketch, line2_idx)
    ix1, iy1 = point_indices(p1)
    ix2, iy2 = point_indices(p2)
    ix3, iy3 = point_indices(p3)
    ix4, iy4 = point_indices(p4)
    rows = [1, 1, 1, 1, 1, 1, 1, 1]
    cols = [ix1, iy1, ix2, iy2, ix3, iy3, ix4, iy4]
    residual! = function (out, x, offset)
        dx12 = x[ix2] - x[ix1]
        dy12 = x[iy2] - x[iy1]
        dx34 = x[ix4] - x[ix3]
        dy34 = x[iy4] - x[iy3]
        out[offset + 1] = dx12 * dy34 - dy12 * dx34
        return nothing
    end
    jacobian! = function (J, x, offset)
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
    return add_constraint!(sketch, Constraint(1, rows, cols, residual!, jacobian!))
end

"""
    build_problem!(sketch)

Construct a SparseLNNS problem from current constraints.
"""
function build_problem!(sketch::Sketch{T}) where {T<:AbstractFloat}
    m = sketch.m
    n = length(sketch.x)
    m > 0 || throw(ArgumentError("cannot build problem with no constraints"))
    n > 0 || throw(ArgumentError("cannot build problem with no points"))

    Jpat = spzeros(m, n)
    offset = 0
    for c in sketch.constraints
        for k in eachindex(c.rows)
            Jpat[offset + c.rows[k], c.cols[k]] = 1.0
        end
        offset += c.m
    end

    r! = function (out, x)
        fill!(out, zero(eltype(out)))
        offset = 0
        for c in sketch.constraints
            c.residual!(out, x, offset)
            offset += c.m
        end
        return nothing
    end

    J! = function (J, x)
        fill!(nonzeros(J), zero(eltype(nonzeros(J))))
        offset = 0
        for c in sketch.constraints
            c.jacobian!(J, x, offset)
            offset += c.m
        end
        return nothing
    end

    sketch.problem = SparseLNNS.Problem(r!, J!, Jpat)
    sketch.structure_dirty = false
    return sketch.problem
end

"""
    solve!(sketch; options=SparseLNNS.Options())

Solve the current constraint system and update `sketch.x` in place.
"""
function solve!(sketch::Sketch{T}; options::SparseLNNS.Options{T} = SparseLNNS.Options{T}()) where {T<:AbstractFloat}
    if sketch.problem === nothing || sketch.structure_dirty
        build_problem!(sketch)
        sketch.state, sketch.work = SparseLNNS.initialize(sketch.problem, sketch.x; options=options)
    else
        copyto!(sketch.state.x, sketch.x)
    end
    stats = SparseLNNS.solve!(sketch.state, sketch.problem, sketch.work; options=options)
    copyto!(sketch.x, sketch.state.x)
    return stats
end

end # module CADConstraints
