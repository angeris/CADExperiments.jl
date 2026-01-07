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
    dxr = x[ix2] - x[ix1]
    dyr = x[iy2] - x[iy1]
    dxp = x[ixp] - x[ix1]
    dyp = x[iyp] - x[iy1]
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
