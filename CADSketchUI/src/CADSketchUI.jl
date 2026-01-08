module CADSketchUI

import CImGui as ig
import GLFW
import ModernGL
import CADConstraints
import Printf: @sprintf

export run

@inline function dist2(a::ig.ImVec2, b::ig.ImVec2)
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy
end

@inline function world_to_screen(p, origin::ig.ImVec2, center, scale::Cfloat)
    return ig.ImVec2(
        origin.x + Cfloat(p[1] - center[1]) * scale,
        origin.y - Cfloat(p[2] - center[2]) * scale,
    )
end

@inline function screen_to_world(p::ig.ImVec2, origin::ig.ImVec2, center, scale::Cfloat)
    x = (p.x - origin.x) / scale + center[1]
    y = -(p.y - origin.y) / scale + center[2]
    return (x, y)
end

@inline function point_count(sketch)
    return length(sketch.x) ÷ 2
end

@inline function point_xy(sketch, p)
    i = 2 * (p - 1) + 1
    return (sketch.x[i], sketch.x[i + 1])
end

struct Measurement
    p1::Int
    p2::Int
    offset::Float64
end

mutable struct AppState
    sketch
    measurements::Vector{Measurement}
    selected::Int
    hovered::Int
    dragging::Int
    tool::Symbol
    line_start::Int
    circle_center::Int
    measure_start::Int
    measure_selected::Int
    measure_hovered::Int
    measure_dragging::Int
    measure_label_hot::Int
    measure_editing::Int
    constraint_selected::Int
    measure_edit_buf::Vector{UInt8}
    measure_edit_focus::Bool
    measure_edit_started::Bool
    center::NTuple{2, Float64}
    scale::Float64
    stats
    report
    residuals::Vector{Float64}
    solve_time_ms::Float64
end

@inline function measurement_offset_delta(p1, p2, scale_f, mouse_delta)
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    len = sqrt(dx * dx + dy * dy)
    if len <= 1e-9
        return 0.0
    end
    nx = -dy / len
    ny = dx / len
    nsx = nx * scale_f
    nsy = -ny * scale_f
    nlen = sqrt(nsx * nsx + nsy * nsy)
    if nlen <= 1e-6
        return 0.0
    end
    nsx /= nlen
    nsy /= nlen
    return (mouse_delta.x * nsx + mouse_delta.y * nsy) / scale_f
end

@inline function set_edit_buffer!(buf, text)
    fill!(buf, 0x00)
    bytes = codeunits(text)
    n = min(length(bytes), length(buf) - 1)
    @inbounds for i in 1:n
        buf[i] = bytes[i]
    end
    return nothing
end

@inline function buffer_string(buf)
    n = findfirst(==(0x00), buf)
    n === nothing && (n = length(buf) + 1)
    return String(buf[1:(n - 1)])
end

@inline function dist2_point_segment(p::ig.ImVec2, a::ig.ImVec2, b::ig.ImVec2)
    vx = b.x - a.x
    vy = b.y - a.y
    denom = vx * vx + vy * vy
    if denom <= 1.0f-9
        dx = p.x - a.x
        dy = p.y - a.y
        return dx * dx + dy * dy
    end
    t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / denom
    if t < 0.0f0
        t = 0.0f0
    elseif t > 1.0f0
        t = 1.0f0
    end
    projx = a.x + t * vx
    projy = a.y + t * vy
    dx = p.x - projx
    dy = p.y - projy
    return dx * dx + dy * dy
end

@inline function dimension_line_points(p1, p2, origin, center, scale_f, offset)
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    len = sqrt(dx * dx + dy * dy)
    if len <= 1e-9
        return nothing
    end
    ux = dx / len
    uy = dy / len
    nx = -uy
    ny = ux
    p1o = (p1[1] + nx * offset, p1[2] + ny * offset)
    p2o = (p2[1] + nx * offset, p2[2] + ny * offset)
    s1o = world_to_screen(p1o, origin, center, scale_f)
    s2o = world_to_screen(p2o, origin, center, scale_f)
    return s1o, s2o
end

function draw_dimension!(draw_list, p1, p2, origin, center, scale_f, color, offset)
    dx = p2[1] - p1[1]
    dy = p2[2] - p1[2]
    len = sqrt(dx * dx + dy * dy)
    if len <= 1e-9
        return
    end
    ux = dx / len
    uy = dy / len
    nx = -uy
    ny = ux
    s1 = world_to_screen(p1, origin, center, scale_f)
    s2 = world_to_screen(p2, origin, center, scale_f)
    pts = dimension_line_points(p1, p2, origin, center, scale_f, offset)
    pts === nothing && return
    s1o, s2o = pts

    ig.AddLine(draw_list, s1, s1o, color, 1.0f0)
    ig.AddLine(draw_list, s2, s2o, color, 1.0f0)

    dxs = s2o.x - s1o.x
    dys = s2o.y - s1o.y
    len_s = sqrt(dxs * dxs + dys * dys)
    if len_s <= 1e-6
        return
    end
    uxs = dxs / len_s
    uys = dys / len_s
    pxs = -uys
    pys = uxs
    arrow_len = 8.0f0
    arrow_w = 4.0f0

    base1 = ig.ImVec2(s1o.x + uxs * arrow_len, s1o.y + uys * arrow_len)
    left1 = ig.ImVec2(base1.x + pxs * arrow_w, base1.y + pys * arrow_w)
    right1 = ig.ImVec2(base1.x - pxs * arrow_w, base1.y - pys * arrow_w)
    ig.AddLine(draw_list, s1o, left1, color, 1.0f0)
    ig.AddLine(draw_list, s1o, right1, color, 1.0f0)

    base2 = ig.ImVec2(s2o.x - uxs * arrow_len, s2o.y - uys * arrow_len)
    left2 = ig.ImVec2(base2.x + pxs * arrow_w, base2.y + pys * arrow_w)
    right2 = ig.ImVec2(base2.x - pxs * arrow_w, base2.y - pys * arrow_w)
    ig.AddLine(draw_list, s2o, left2, color, 1.0f0)
    ig.AddLine(draw_list, s2o, right2, color, 1.0f0)

    label = @sprintf("%.3f", len)
    text_size = ig.CalcTextSize(label)
    mid = ig.ImVec2((s1o.x + s2o.x) * 0.5f0, (s1o.y + s2o.y) * 0.5f0)
    text_pos = ig.ImVec2(mid.x - text_size.x * 0.5f0, mid.y - text_size.y * 0.5f0)
    gap = text_size.x * 0.5f0 + 6.0f0
    if len_s > 2.0f0 * gap
        left = ig.ImVec2(mid.x - uxs * gap, mid.y - uys * gap)
        right = ig.ImVec2(mid.x + uxs * gap, mid.y + uys * gap)
        ig.AddLine(draw_list, s1o, left, color, 1.5f0)
        ig.AddLine(draw_list, right, s2o, color, 1.5f0)
    else
        ig.AddLine(draw_list, s1o, s2o, color, 1.5f0)
    end
    ig.AddText(draw_list, text_pos, color, label)
    return nothing
end

@inline function constraint_residuals!(out, sketch)
    resize!(out, length(sketch.constraints))
    if sketch.value_dirty
        sketch.problem.r!(sketch.work.r, sketch.x)
    end
    offset = 0
    for (idx, constraint) in enumerate(sketch.constraints)
        rows = CADConstraints.constraint_rows(constraint)
        sumsq = 0.0
        @inbounds for j in 1:rows
            r = sketch.work.r[offset + j]
            sumsq += r * r
        end
        offset += rows
        out[idx] = sqrt(sumsq)
    end
    return nothing
end

@inline function solve_and_update!(app)
    t0 = time_ns()
    app.stats = CADConstraints.solve!(app.sketch)
    app.solve_time_ms = (time_ns() - t0) / 1e6
    app.report = CADConstraints.conflicts(app.sketch, app.stats)
    constraint_residuals!(app.residuals, app.sketch)
    return nothing
end

@inline function delete_selected_constraint!(app)
    idx = app.constraint_selected
    if idx == 0
        return nothing
    end
    if length(app.sketch.constraints) <= 1
        return nothing
    end
    constraint = app.sketch.constraints[idx]
    if constraint isa CADConstraints.FixedPoint
        if constraint.p == 1 && constraint.x == 0.0 && constraint.y == 0.0
            return nothing
        end
    end
    deleteat!(app.sketch.constraints, idx)
    CADConstraints.mark_structure_dirty!(app.sketch)
    app.constraint_selected = 0
    solve_and_update!(app)
    return nothing
end

@inline function handle_constraint_delete!(app)
    if app.constraint_selected == 0
        return nothing
    end
    io = unsafe_load(ig.GetIO())
    if io.WantTextInput
        return nothing
    end
    if ig.IsKeyPressed(ig.ImGuiKey_Delete, false) || ig.IsKeyPressed(ig.ImGuiKey_Backspace, false)
        delete_selected_constraint!(app)
    end
    return nothing
end

@inline function color_u32(r::Float32, g::Float32, b::Float32, a::Float32 = 1.0f0)
    return ig.GetColorU32(ig.ImVec4(r, g, b, a))
end

function draw_sketch!(app)
    sketch = app.sketch
    measurements = app.measurements

    vp = unsafe_load(ig.GetMainViewport())
    ig.SetNextWindowPos((vp.Pos.x, vp.Pos.y))
    ig.SetNextWindowSize((vp.Size.x, vp.Size.y))
    flags = ig.ImGuiWindowFlags_NoTitleBar |
            ig.ImGuiWindowFlags_NoResize |
            ig.ImGuiWindowFlags_NoMove |
            ig.ImGuiWindowFlags_NoScrollbar |
            ig.ImGuiWindowFlags_NoScrollWithMouse |
            ig.ImGuiWindowFlags_NoCollapse |
            ig.ImGuiWindowFlags_NoBringToFrontOnFocus |
            ig.ImGuiWindowFlags_NoNavFocus
    ig.Begin("Sketch", C_NULL, flags)

    canvas_size = ig.GetContentRegionAvail()
    if canvas_size.x < 50 || canvas_size.y < 50
        ig.TextUnformatted("Resize window to view canvas.")
        ig.End()
        return
    end

    canvas_pos = ig.GetCursorScreenPos()
    ig.InvisibleButton("canvas", canvas_size)
    is_hovered = ig.IsItemHovered()

    draw_list = ig.GetWindowDrawList()
    p_min = canvas_pos
    p_max = ig.ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y)
    ig.AddRectFilled(draw_list, p_min, p_max, color_u32(0.08f0, 0.08f0, 0.08f0))
    ig.AddRect(draw_list, p_min, p_max, color_u32(0.20f0, 0.20f0, 0.20f0))

    scale_f = Cfloat(app.scale)
    origin = ig.ImVec2(canvas_pos.x + canvas_size.x * 0.5f0, canvas_pos.y + canvas_size.y * 0.5f0)
    mouse = ig.GetMousePos()
    world_mouse = screen_to_world(mouse, origin, app.center, scale_f)
    io = unsafe_load(ig.GetIO())
    if !io.WantTextInput && ig.IsKeyPressed(ig.ImGuiKey_Escape, false)
        app.tool = :select
        reset_tool_state!(app)
        ig.SetWindowFocus("Sketch")
    end

    if is_hovered && ig.IsMouseDown(1)
        dx = io.MouseDelta.x
        dy = io.MouseDelta.y
        if dx != 0 || dy != 0
            app.center = (app.center[1] - dx / scale_f, app.center[2] + dy / scale_f)
        end
    end

    if is_hovered && io.MouseWheel != 0
        world_before = screen_to_world(mouse, origin, app.center, scale_f)
        factor = 1.1f0 ^ io.MouseWheel
        scale_new = clamp(scale_f * factor, 10.0f0, 500.0f0)
        world_after = screen_to_world(mouse, origin, app.center, scale_new)
        app.center = (app.center[1] + world_before[1] - world_after[1],
                      app.center[2] + world_before[2] - world_after[2])
        app.scale = scale_new
        scale_f = scale_new
    end

    app.hovered = 0
    hit_radius = 8.0f0
    for idx in 1:point_count(sketch)
        sp = world_to_screen(point_xy(sketch, idx), origin, app.center, scale_f)
        if dist2(sp, mouse) <= hit_radius * hit_radius
            app.hovered = idx
            break
        end
    end

    line_hovered = 0
    line_best = hit_radius * hit_radius
    for (idx, line) in enumerate(sketch.lines)
        sp1 = world_to_screen(point_xy(sketch, line.p1), origin, app.center, scale_f)
        sp2 = world_to_screen(point_xy(sketch, line.p2), origin, app.center, scale_f)
        d2 = dist2_point_segment(mouse, sp1, sp2)
        if d2 <= line_best
            line_best = d2
            line_hovered = idx
        end
    end

    circle_hovered = 0
    circle_best = hit_radius
    for (idx, circle) in enumerate(sketch.circles)
        spc = world_to_screen(point_xy(sketch, circle.center), origin, app.center, scale_f)
        spr = world_to_screen(point_xy(sketch, circle.rim), origin, app.center, scale_f)
        dxr = spr.x - spc.x
        dyr = spr.y - spc.y
        r = sqrt(dxr * dxr + dyr * dyr)
        d = sqrt(dist2(mouse, spc))
        if r > 1.0f-6
            dr = abs(d - r)
            if dr <= circle_best
                circle_best = dr
                circle_hovered = idx
            end
        end
    end

    app.measure_hovered = 0
    app.measure_label_hot = 0
    best_d2 = 64.0f0
    for (idx, measurement) in enumerate(measurements)
        p1 = point_xy(sketch, measurement.p1)
        p2 = point_xy(sketch, measurement.p2)
        pts = dimension_line_points(p1, p2, origin, app.center, scale_f, measurement.offset)
        pts === nothing && continue
        s1o, s2o = pts
        d2 = dist2_point_segment(mouse, s1o, s2o)
        dx = p2[1] - p1[1]
        dy = p2[2] - p1[2]
        len = sqrt(dx * dx + dy * dy)
        if len > 1e-9
            label = @sprintf("%.3f", len)
            text_size = ig.CalcTextSize(label)
            mid = ig.ImVec2((s1o.x + s2o.x) * 0.5f0, (s1o.y + s2o.y) * 0.5f0)
            minx = mid.x - text_size.x * 0.5f0
            maxx = mid.x + text_size.x * 0.5f0
            miny = mid.y - text_size.y * 0.5f0
            maxy = mid.y + text_size.y * 0.5f0
            if mouse.x >= minx && mouse.x <= maxx && mouse.y >= miny && mouse.y <= maxy
                d2 = 0.0f0
                app.measure_label_hot = idx
            end
        end
        if d2 < best_d2
            best_d2 = d2
            app.measure_hovered = idx
        end
    end

    handled_click = false
    if is_hovered && ig.IsMouseDoubleClicked(0) && app.tool == :select && app.measure_label_hot != 0
        app.measure_editing = app.measure_label_hot
        app.measure_selected = app.measure_label_hot
        app.measure_dragging = 0
        app.selected = 0
        app.dragging = 0
        m = measurements[app.measure_editing]
        dx = point_xy(sketch, m.p2)[1] - point_xy(sketch, m.p1)[1]
        dy = point_xy(sketch, m.p2)[2] - point_xy(sketch, m.p1)[2]
        len = sqrt(dx * dx + dy * dy)
        set_edit_buffer!(app.measure_edit_buf, @sprintf("%.3f", len))
        app.measure_edit_focus = true
        app.measure_edit_started = true
        handled_click = true
    end

    if is_hovered && ig.IsMouseClicked(0) && !handled_click
        wx, wy = screen_to_world(mouse, origin, app.center, scale_f)
        if app.tool == :select
            if app.measure_hovered != 0
                app.measure_selected = app.measure_hovered
                app.measure_dragging = app.measure_hovered
                app.selected = 0
                app.dragging = 0
            else
                app.measure_selected = 0
                app.measure_dragging = 0
                app.selected = app.hovered
                app.dragging = app.hovered == 1 ? 0 : app.hovered
            end
        elseif app.tool == :point
            p = CADConstraints.add_point!(sketch, wx, wy)
            app.selected = p
            solve_and_update!(app)
        elseif app.tool == :line
            if app.line_start == 0
                app.line_start = app.hovered != 0 ? app.hovered : CADConstraints.add_point!(sketch, wx, wy)
            else
                p2 = app.hovered != 0 ? app.hovered : CADConstraints.add_point!(sketch, wx, wy)
                push!(sketch, CADConstraints.Line(app.line_start, p2))
                app.selected = p2
                app.line_start = 0
                solve_and_update!(app)
            end
        elseif app.tool == :circle
            if app.circle_center == 0
                app.circle_center = app.hovered != 0 ? app.hovered : CADConstraints.add_point!(sketch, wx, wy)
            else
                rim = app.hovered != 0 ? app.hovered : CADConstraints.add_point!(sketch, wx, wy)
                push!(sketch, CADConstraints.Circle(app.circle_center, rim))
                app.selected = rim
                app.circle_center = 0
                solve_and_update!(app)
            end
        elseif app.tool == :measure
            if app.hovered == 0 && line_hovered == 0 && circle_hovered == 0
                app.measure_start = 0
            elseif app.hovered != 0
                if app.measure_start == 0
                    app.measure_start = app.hovered
                elseif app.hovered != app.measure_start
                    offset = 20.0 / scale_f
                    push!(measurements, Measurement(app.measure_start, app.hovered, offset))
                    app.selected = app.hovered
                    app.measure_start = 0
                end
            elseif line_hovered != 0
                line = sketch.lines[line_hovered]
                offset = 20.0 / scale_f
                push!(measurements, Measurement(line.p1, line.p2, offset))
                app.measure_start = 0
            elseif circle_hovered != 0
                circle = sketch.circles[circle_hovered]
                offset = 20.0 / scale_f
                push!(measurements, Measurement(circle.center, circle.rim, offset))
                app.measure_start = 0
            end
        end
    end
    if app.tool == :select && app.measure_dragging != 0 && app.measure_editing == 0
        if ig.IsMouseDown(0)
            if io.MouseDelta.x != 0 || io.MouseDelta.y != 0
                m = measurements[app.measure_dragging]
                delta = measurement_offset_delta(point_xy(sketch, m.p1), point_xy(sketch, m.p2), scale_f, io.MouseDelta)
                measurements[app.measure_dragging] = Measurement(m.p1, m.p2, m.offset + delta)
            end
        else
            app.measure_dragging = 0
        end
    end
    if app.tool == :select && app.dragging != 0 && app.measure_editing == 0
        if app.dragging == 1
            app.dragging = 0
        elseif ig.IsMouseDown(0)
            if io.MouseDelta.x != 0 || io.MouseDelta.y != 0
                wx, wy = screen_to_world(mouse, origin, app.center, scale_f)
                CADConstraints.set_point!(sketch, app.dragging, wx, wy)
                solve_and_update!(app)
            end
        else
            app.dragging = 0
        end
    end

    for line in sketch.lines
        p1 = world_to_screen(point_xy(sketch, line.p1), origin, app.center, scale_f)
        p2 = world_to_screen(point_xy(sketch, line.p2), origin, app.center, scale_f)
        ig.AddLine(draw_list, p1, p2, color_u32(0.55f0, 0.65f0, 0.75f0), 2.0f0)
    end

    for circle in sketch.circles
        c = world_to_screen(point_xy(sketch, circle.center), origin, app.center, scale_f)
        r = world_to_screen(point_xy(sketch, circle.rim), origin, app.center, scale_f)
        dx = r.x - c.x
        dy = r.y - c.y
        radius = sqrt(dx * dx + dy * dy)
        if radius > 0.5f0
            ig.AddCircle(draw_list, c, radius, color_u32(0.55f0, 0.65f0, 0.75f0), 64, 2.0f0)
        end
    end

    if app.tool == :line && app.line_start != 0
        p1 = world_to_screen(point_xy(sketch, app.line_start), origin, app.center, scale_f)
        p2 = ig.GetMousePos()
        ig.AddLine(draw_list, p1, p2, color_u32(0.30f0, 0.80f0, 0.90f0), 2.0f0)
    elseif app.tool == :circle && app.circle_center != 0
        center_p = world_to_screen(point_xy(sketch, app.circle_center), origin, app.center, scale_f)
        dx = mouse.x - center_p.x
        dy = mouse.y - center_p.y
        r = sqrt(dx * dx + dy * dy)
        ig.AddCircle(draw_list, center_p, r, color_u32(0.30f0, 0.80f0, 0.90f0), 64, 2.0f0)
    end

        for (idx, measurement) in enumerate(measurements)
            active = idx == app.measure_selected || idx == app.measure_hovered
            color = active ? color_u32(0.95f0, 0.80f0, 0.25f0) : color_u32(0.85f0, 0.85f0, 0.25f0)
            draw_dimension!(draw_list, point_xy(sketch, measurement.p1), point_xy(sketch, measurement.p2),
                            origin, app.center, scale_f, color, measurement.offset)
        end
        if app.measure_editing != 0
            m = measurements[app.measure_editing]
            p1 = point_xy(sketch, m.p1)
            p2 = point_xy(sketch, m.p2)
            pts = dimension_line_points(p1, p2, origin, app.center, scale_f, m.offset)
            if pts !== nothing
                s1o, s2o = pts
                dx = p2[1] - p1[1]
                dy = p2[2] - p1[2]
                len = sqrt(dx * dx + dy * dy)
                label = @sprintf("%.3f", len)
                text_size = ig.CalcTextSize(label)
                mid = ig.ImVec2((s1o.x + s2o.x) * 0.5f0, (s1o.y + s2o.y) * 0.5f0)
                text_pos = ig.ImVec2(mid.x - text_size.x * 0.5f0, mid.y - text_size.y * 0.5f0)
                ig.SetCursorScreenPos(text_pos)
                ig.SetNextItemWidth(text_size.x + 16.0f0)
                if app.measure_edit_focus
                    ig.SetKeyboardFocusHere()
                    app.measure_edit_focus = false
                end
                flags = ig.ImGuiInputTextFlags_EnterReturnsTrue |
                        ig.ImGuiInputTextFlags_AutoSelectAll |
                        ig.ImGuiInputTextFlags_CharsDecimal
                submitted = ig.InputText(@sprintf("##measure_edit_%d", app.measure_editing), app.measure_edit_buf, length(app.measure_edit_buf), flags)
                if submitted
                    value = tryparse(Float64, strip(buffer_string(app.measure_edit_buf)))
                    if value !== nothing && value > 0
                        push!(sketch, CADConstraints.Distance(m.p1, m.p2, value))
                        solve_and_update!(app)
                    end
                    app.measure_editing = 0
                elseif !app.measure_edit_started && !ig.IsItemActive() && ig.IsMouseClicked(0)
                    app.measure_editing = 0
                end
                app.measure_edit_started = false
            end
        end
    if app.tool == :measure && app.measure_start != 0
        preview_offset = 20.0 / scale_f
        draw_dimension!(draw_list, point_xy(sketch, app.measure_start), world_mouse,
                        origin, app.center, scale_f, color_u32(0.85f0, 0.85f0, 0.25f0),
                        preview_offset)
    end

    for idx in 1:point_count(sketch)
        sp = world_to_screen(point_xy(sketch, idx), origin, app.center, scale_f)
        if idx == 1
            if idx == app.selected || idx == app.hovered
                color = color_u32(0.98f0, 0.55f0, 0.55f0)
            else
                color = color_u32(0.90f0, 0.30f0, 0.30f0)
            end
            ig.AddCircleFilled(draw_list, sp, 5.0f0, color)
            cross = 6.0f0
            ig.AddLine(draw_list, ig.ImVec2(sp.x - cross, sp.y), ig.ImVec2(sp.x + cross, sp.y), color, 1.5f0)
            ig.AddLine(draw_list, ig.ImVec2(sp.x, sp.y - cross), ig.ImVec2(sp.x, sp.y + cross), color, 1.5f0)
        else
            if idx == app.selected
                color = color_u32(0.95f0, 0.70f0, 0.20f0)
            elseif idx == app.hovered
                color = color_u32(0.80f0, 0.80f0, 0.25f0)
            else
                color = color_u32(0.90f0, 0.90f0, 0.90f0)
            end
            ig.AddCircleFilled(draw_list, sp, 4.0f0, color)
        end
    end

    ig.End()
end

function tool_button(label, key, app, size)
    active = app.tool == key
    if active
        ig.PushStyleColor(ig.ImGuiCol_Button, ig.ImVec4(0.25f0, 0.55f0, 0.85f0, 1.0f0))
        ig.PushStyleColor(ig.ImGuiCol_ButtonHovered, ig.ImVec4(0.30f0, 0.60f0, 0.90f0, 1.0f0))
        ig.PushStyleColor(ig.ImGuiCol_ButtonActive, ig.ImVec4(0.20f0, 0.50f0, 0.80f0, 1.0f0))
    end
    clicked = ig.Button(label, size)
    if active
        ig.PopStyleColor(3)
    end
    if clicked
        app.tool = key
    end
    return clicked
end

function reset_tool_state!(app)
    app.line_start = 0
    app.circle_center = 0
    app.measure_start = 0
    app.measure_dragging = 0
    app.measure_editing = 0
    app.measure_label_hot = 0
    app.measure_edit_started = false
    app.dragging = 0
    return nothing
end

function draw_toolbar!(app)
    ig.SetNextWindowPos((10.0f0, 10.0f0), ig.ImGuiCond_FirstUseEver)
    ig.SetNextWindowSize((180.0f0, 0.0f0), ig.ImGuiCond_FirstUseEver)
    flags = ig.ImGuiWindowFlags_AlwaysAutoResize
    ig.Begin("Tools", C_NULL, flags)
    labels = ("Select", "Point", "Line", "Circle", "Measure")
    max_w = 0.0f0
    max_h = 0.0f0
    for label in labels
        sz = ig.CalcTextSize(label)
        max_w = max(max_w, sz.x)
        max_h = max(max_h, sz.y)
    end
    style = unsafe_load(ig.GetStyle())
    pad = style.FramePadding
    button_w = max_w + 2.0f0 * pad.x
    button_h = max_h + 2.0f0 * pad.y
    button_size = (button_w, button_h)

    ig.PushStyleVar(ig.ImGuiStyleVar_ButtonTextAlign, ig.ImVec2(0.5f0, 0.5f0))
    if ig.BeginTable("tools_table", 2, ig.ImGuiTableFlags_SizingFixedFit)
        ig.TableSetupColumn("col1", ig.ImGuiTableColumnFlags_WidthFixed, button_w)
        ig.TableSetupColumn("col2", ig.ImGuiTableColumnFlags_WidthFixed, button_w)
        ig.TableNextColumn()
        changed = tool_button("Select", :select, app, button_size)
        ig.TableNextColumn()
        changed = tool_button("Point", :point, app, button_size) || changed
        ig.TableNextColumn()
        changed = tool_button("Line", :line, app, button_size) || changed
        ig.TableNextColumn()
        changed = tool_button("Circle", :circle, app, button_size) || changed
        ig.TableNextColumn()
        changed = tool_button("Measure", :measure, app, button_size) || changed
        ig.EndTable()
        if changed
            reset_tool_state!(app)
        end
    end
    ig.PopStyleVar()
    ig.End()
end

function draw_status_panel!(app)
    stats = app.stats
    report = app.report
    vp = unsafe_load(ig.GetMainViewport())
    padding = 12.0f0
    ig.SetNextWindowPos((vp.Pos.x + vp.Size.x - padding, vp.Pos.y + vp.Size.y - padding),
                        ig.ImGuiCond_Always, (1.0f0, 1.0f0))
    flags = ig.ImGuiWindowFlags_AlwaysAutoResize |
            ig.ImGuiWindowFlags_NoResize |
            ig.ImGuiWindowFlags_NoMove |
            ig.ImGuiWindowFlags_NoCollapse |
            ig.ImGuiWindowFlags_NoTitleBar
    ig.Begin("Status", C_NULL, flags)

    converged = stats.status == :converged
    conflicted = report.conflicted
    if converged && conflicted
        color = ig.ImVec4(0.95f0, 0.65f0, 0.20f0, 1.0f0)
        label = "Converged (conflict)"
    elseif converged
        color = ig.ImVec4(0.20f0, 0.80f0, 0.35f0, 1.0f0)
        label = "Converged"
    else
        color = ig.ImVec4(0.90f0, 0.25f0, 0.25f0, 1.0f0)
        label = "Not Converged"
    end
    ig.TextColored(color, label)
    ig.TextUnformatted(@sprintf("iters: %d", stats.iters))
    ig.TextUnformatted(@sprintf("cost: %.3e", stats.cost))
    ig.TextUnformatted(@sprintf("residual: %.3e", report.residual_norm))
    ig.TextUnformatted(@sprintf("solve: %.2f ms", app.solve_time_ms))

    if report.conflicted && !isempty(report.entries)
        ig.Separator()
        ig.TextUnformatted("Conflicting constraints")
        for entry in report.entries
            ig.TextUnformatted(@sprintf("%s #%d: %.3e", String(entry.kind), entry.index, entry.norm))
        end
    end

    ig.End()
end

@inline function line_has_point(line, p)
    return line.p1 == p || line.p2 == p
end

function constraint_label(constraint, sketch)
    if constraint isa CADConstraints.FixedPoint
        return @sprintf("FixedPoint p%d (%.3f, %.3f)", constraint.p, constraint.x, constraint.y)
    elseif constraint isa CADConstraints.Coincident
        return @sprintf("Coincident p%d-p%d", constraint.p1, constraint.p2)
    elseif constraint isa CADConstraints.Horizontal
        return @sprintf("Horizontal l%d", constraint.line)
    elseif constraint isa CADConstraints.Vertical
        return @sprintf("Vertical l%d", constraint.line)
    elseif constraint isa CADConstraints.Parallel
        return @sprintf("Parallel l%d-l%d", constraint.line1, constraint.line2)
    elseif constraint isa CADConstraints.Distance
        return @sprintf("Distance p%d-p%d (%.3f)", constraint.p1, constraint.p2, constraint.d)
    elseif constraint isa CADConstraints.Diameter
        return @sprintf("Diameter c%d (%.3f)", constraint.circle, constraint.d)
    elseif constraint isa CADConstraints.Normal
        return @sprintf("Normal c%d-l%d", constraint.circle, constraint.line)
    elseif constraint isa CADConstraints.CircleCoincident
        return @sprintf("CircleCoincident c%d p%d", constraint.circle, constraint.p)
    else
        return String(nameof(typeof(constraint)))
    end
end

function draw_selection_panel!(app)
    sketch = app.sketch
    residuals = app.residuals
    vp = unsafe_load(ig.GetMainViewport())
    padding = 12.0f0
    ig.SetNextWindowPos((vp.Pos.x + vp.Size.x - padding, vp.Pos.y + padding),
                        ig.ImGuiCond_Always, (1.0f0, 0.0f0))
    if app.constraint_selected > length(sketch.constraints)
        app.constraint_selected = 0
    end
    max_text_w = 0.0f0
    max_label_w = 0.0f0
    max_res_w = 0.0f0
    if !isempty(residuals)
        for (idx, constraint) in enumerate(sketch.constraints)
            label = @sprintf("%d. %s", idx, constraint_label(constraint, sketch))
            res = @sprintf("r=%.3e", residuals[idx])
            sz = ig.CalcTextSize(label)
            rsz = ig.CalcTextSize(res)
            max_text_w = max(max_text_w, sz.x)
            max_label_w = max(max_label_w, sz.x)
            max_res_w = max(max_res_w, rsz.x)
            max_text_w = max(max_text_w, sz.x + rsz.x + 12.0f0)
        end
    end
    line_h = ig.GetTextLineHeightWithSpacing()
    desired_w = max(max_text_w + 32.0f0, 220.0f0)
    desired_h = clamp((length(sketch.constraints) + 6) * line_h, 160.0f0, 360.0f0)
    ig.SetNextWindowSize((desired_w, desired_h), ig.ImGuiCond_FirstUseEver)
    flags = ig.ImGuiWindowFlags_NoMove
    ig.Begin("Selection", C_NULL, flags)

    if app.selected == 0
        ig.TextUnformatted("Selected: none")
    else
        x, y = point_xy(sketch, app.selected)
        if app.selected == 1
            ig.TextUnformatted("Selected origin (p1)")
        else
            ig.TextUnformatted(@sprintf("Selected p%d", app.selected))
        end
        ig.TextUnformatted(@sprintf("pos: (%.3f, %.3f)", x, y))
        for (idx, line) in enumerate(sketch.lines)
            if line_has_point(line, app.selected)
                ig.TextUnformatted(@sprintf("line l%d: p%d-p%d", idx, line.p1, line.p2))
            end
        end
    end

    ig.Separator()
    if app.measure_selected != 0
        m = app.measurements[app.measure_selected]
        p1 = m.p1
        p2 = m.p2
        x1, y1 = point_xy(sketch, p1)
        x2, y2 = point_xy(sketch, p2)
        dx = x2 - x1
        dy = y2 - y1
        dist = sqrt(dx * dx + dy * dy)
        ig.TextUnformatted(@sprintf("Measurement m%d", app.measure_selected))
        ig.TextUnformatted(@sprintf("points: p%d-p%d", p1, p2))
        ig.TextUnformatted(@sprintf("value: %.3f", dist))
        ig.Separator()
    end
    ig.TextUnformatted("Constraints")
    if ig.BeginChild("constraints", (0.0f0, 0.0f0), true)
        for (idx, constraint) in enumerate(sketch.constraints)
            label = @sprintf("%d. %s", idx, constraint_label(constraint, sketch))
            res = @sprintf("r=%.3e", residuals[idx])
            line_start = ig.GetCursorPosX()
            line_y = ig.GetCursorPosY()
            selected = app.constraint_selected == idx
            if ig.Selectable(label, selected, ig.ImGuiSelectableFlags_SpanAllColumns)
                app.constraint_selected = idx
            end
            res_w = ig.CalcTextSize(res).x
            ig.SameLine()
            ig.SetCursorPosY(line_y)
            res_x = line_start + max_label_w + 12.0f0 + (max_res_w - res_w)
            ig.SetCursorPosX(res_x)
            ig.TextUnformatted(res)
        end
        ig.EndChild()
    end
    if ig.IsWindowFocused(ig.ImGuiFocusedFlags_RootAndChildWindows)
        handle_constraint_delete!(app)
    end

    ig.End()
end

"""
    run(; window_size=(1280, 720), window_title="CADSketchUI")

Open a minimal sketch window with a canvas and a placeholder sketch.
This is the first UI checkpoint (canvas + selection + pan/zoom).
"""
function run(; window_size=(1280, 720), window_title="CADSketchUI")
    ig.set_backend(:GlfwOpenGL3)
    ctx = ig.CreateContext()

    sketch = CADConstraints.Sketch()
    x1, y1 = -2.0, 0.0
    x2, y2 = 2.0, 0.0
    x3, y3 = 2.0, 1.5
    x4, y4 = -2.0, 1.5
    x5, y5 = 2.0, 3.0
    p1 = CADConstraints.add_point!(sketch, x1, y1)
    p2 = CADConstraints.add_point!(sketch, x2, y2)
    p3 = CADConstraints.add_point!(sketch, x3, y3)
    p4 = CADConstraints.add_point!(sketch, x4, y4)
    p5 = CADConstraints.add_point!(sketch, x5, y5)
    l1 = push!(sketch, CADConstraints.Line(p1, p2))
    l2 = push!(sketch, CADConstraints.Line(p2, p3))
    l3 = push!(sketch, CADConstraints.Line(p3, p4))
    l4 = push!(sketch, CADConstraints.Line(p4, p1))
    l5 = push!(sketch, CADConstraints.Line(p3, p5))
    push!(sketch, CADConstraints.Horizontal(l1))
    push!(sketch, CADConstraints.Vertical(l2))
    dx12 = x1 - x2
    dy12 = y1 - y2
    dx23 = x2 - x3
    dy23 = y2 - y3
    dx34 = x3 - x4
    dy34 = y3 - y4
    dx41 = x4 - x1
    dy41 = y4 - y1
    dx35 = x3 - x5
    dy35 = y3 - y5
    d12 = sqrt(dx12 * dx12 + dy12 * dy12)
    d23 = sqrt(dx23 * dx23 + dy23 * dy23)
    d34 = sqrt(dx34 * dx34 + dy34 * dy34)
    d41 = sqrt(dx41 * dx41 + dy41 * dy41)
    d35 = sqrt(dx35 * dx35 + dy35 * dy35)
    push!(sketch, CADConstraints.Distance(p1, p2, d12))
    push!(sketch, CADConstraints.Distance(p2, p3, d23))
    push!(sketch, CADConstraints.Distance(p3, p4, d34))
    push!(sketch, CADConstraints.Distance(p3, p5, d35))
    t0 = time_ns()
    stats = CADConstraints.solve!(sketch)
    solve_time_ms = (time_ns() - t0) / 1e6
    report = CADConstraints.conflicts(sketch, stats)
    residuals = Float64[]
    constraint_residuals!(residuals, sketch)
    @info "Initial solve" iters=stats.iters status=stats.status cost=stats.cost
    for idx in 1:point_count(sketch)
        x, y = point_xy(sketch, idx)
        @info "Point" index=idx x y
    end
    app = AppState(
        sketch,
        Measurement[],
        0,
        0,
        0,
        :select,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        fill(0x00, 32),
        false,
        false,
        (0.0, 0.0),
        80.0,
        stats,
        report,
        residuals,
        solve_time_ms,
    )

    ig.render(ctx; window_size=window_size, window_title=window_title) do
        draw_sketch!(app)
        draw_toolbar!(app)
        draw_status_panel!(app)
        draw_selection_panel!(app)
    end
    return nothing
end

end # module CADSketchUI
