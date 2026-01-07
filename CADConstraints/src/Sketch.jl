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
