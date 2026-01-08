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

@inline function solve_and_update!(sketch, stats_ref, report_ref, residuals_ref, solve_time_ms)
    t0 = time_ns()
    stats_ref[] = CADConstraints.solve!(sketch)
    solve_time_ms[] = (time_ns() - t0) / 1e6
    report_ref[] = CADConstraints.conflicts(sketch, stats_ref[])
    constraint_residuals!(residuals_ref[], sketch)
    return nothing
end

@inline function color_u32(r::Float32, g::Float32, b::Float32, a::Float32 = 1.0f0)
    return ig.GetColorU32(ig.ImVec4(r, g, b, a))
end

function draw_sketch!(sketch, selected, hovered, dragging, tool, line_start, circle_center, center, scale, stats_ref, report_ref, residuals_ref, solve_time_ms)
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

    scale_f = Cfloat(scale[])
    origin = ig.ImVec2(canvas_pos.x + canvas_size.x * 0.5f0, canvas_pos.y + canvas_size.y * 0.5f0)
    mouse = ig.GetMousePos()
    io = unsafe_load(ig.GetIO())

    if is_hovered && ig.IsMouseDown(1)
        dx = io.MouseDelta.x
        dy = io.MouseDelta.y
        if dx != 0 || dy != 0
            center[] = (center[][1] - dx / scale_f, center[][2] + dy / scale_f)
        end
    end

    if is_hovered && io.MouseWheel != 0
        world_before = screen_to_world(mouse, origin, center[], scale_f)
        factor = 1.1f0 ^ io.MouseWheel
        scale_new = clamp(scale_f * factor, 10.0f0, 500.0f0)
        world_after = screen_to_world(mouse, origin, center[], scale_new)
        center[] = (center[][1] + world_before[1] - world_after[1],
                    center[][2] + world_before[2] - world_after[2])
        scale[] = scale_new
        scale_f = scale_new
    end

    hovered[] = 0
    hit_radius = 8.0f0
    for idx in 1:point_count(sketch)
        sp = world_to_screen(point_xy(sketch, idx), origin, center[], scale_f)
        if dist2(sp, mouse) <= hit_radius * hit_radius
            hovered[] = idx
            break
        end
    end

    if is_hovered && ig.IsMouseClicked(0)
        wx, wy = screen_to_world(mouse, origin, center[], scale_f)
        if tool[] == :select
            selected[] = hovered[]
            dragging[] = hovered[]
        elseif tool[] == :point
            p = CADConstraints.add_point!(sketch, wx, wy)
            selected[] = p
            solve_and_update!(sketch, stats_ref, report_ref, residuals_ref, solve_time_ms)
        elseif tool[] == :line
            if line_start[] == 0
                line_start[] = hovered[] != 0 ? hovered[] : CADConstraints.add_point!(sketch, wx, wy)
            else
                p2 = hovered[] != 0 ? hovered[] : CADConstraints.add_point!(sketch, wx, wy)
                push!(sketch, CADConstraints.Line(line_start[], p2))
                selected[] = p2
                line_start[] = 0
                solve_and_update!(sketch, stats_ref, report_ref, residuals_ref, solve_time_ms)
            end
        elseif tool[] == :circle
            if circle_center[] == 0
                circle_center[] = hovered[] != 0 ? hovered[] : CADConstraints.add_point!(sketch, wx, wy)
            else
                rim = hovered[] != 0 ? hovered[] : CADConstraints.add_point!(sketch, wx, wy)
                push!(sketch, CADConstraints.Circle(circle_center[], rim))
                selected[] = rim
                circle_center[] = 0
                solve_and_update!(sketch, stats_ref, report_ref, residuals_ref, solve_time_ms)
            end
        end
    end
    if tool[] == :select && dragging[] != 0
        if ig.IsMouseDown(0)
            if io.MouseDelta.x != 0 || io.MouseDelta.y != 0
                wx, wy = screen_to_world(mouse, origin, center[], scale_f)
                CADConstraints.set_point!(sketch, dragging[], wx, wy)
                solve_and_update!(sketch, stats_ref, report_ref, residuals_ref, solve_time_ms)
            end
        else
            dragging[] = 0
        end
    end

    for line in sketch.lines
        p1 = world_to_screen(point_xy(sketch, line.p1), origin, center[], scale_f)
        p2 = world_to_screen(point_xy(sketch, line.p2), origin, center[], scale_f)
        ig.AddLine(draw_list, p1, p2, color_u32(0.55f0, 0.65f0, 0.75f0), 2.0f0)
    end

    for circle in sketch.circles
        c = world_to_screen(point_xy(sketch, circle.center), origin, center[], scale_f)
        r = world_to_screen(point_xy(sketch, circle.rim), origin, center[], scale_f)
        dx = r.x - c.x
        dy = r.y - c.y
        radius = sqrt(dx * dx + dy * dy)
        if radius > 0.5f0
            ig.AddCircle(draw_list, c, radius, color_u32(0.55f0, 0.65f0, 0.75f0), 64, 2.0f0)
        end
    end

    if tool[] == :line && line_start[] != 0
        p1 = world_to_screen(point_xy(sketch, line_start[]), origin, center[], scale_f)
        p2 = ig.GetMousePos()
        ig.AddLine(draw_list, p1, p2, color_u32(0.30f0, 0.80f0, 0.90f0), 2.0f0)
    elseif tool[] == :circle && circle_center[] != 0
        center_p = world_to_screen(point_xy(sketch, circle_center[]), origin, center[], scale_f)
        dx = mouse.x - center_p.x
        dy = mouse.y - center_p.y
        r = sqrt(dx * dx + dy * dy)
        ig.AddCircle(draw_list, center_p, r, color_u32(0.30f0, 0.80f0, 0.90f0), 64, 2.0f0)
    end

    for idx in 1:point_count(sketch)
        sp = world_to_screen(point_xy(sketch, idx), origin, center[], scale_f)
        if idx == selected[]
            color = color_u32(0.95f0, 0.70f0, 0.20f0)
        elseif idx == hovered[]
            color = color_u32(0.80f0, 0.80f0, 0.25f0)
        else
            color = color_u32(0.90f0, 0.90f0, 0.90f0)
        end
        ig.AddCircleFilled(draw_list, sp, 4.0f0, color)
    end

    ig.End()
end

function tool_button(label, key, tool, size)
    active = tool[] == key
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
        tool[] = key
    end
    return clicked
end

function draw_toolbar!(tool, line_start, circle_center, dragging)
    ig.SetNextWindowPos((10.0f0, 10.0f0), ig.ImGuiCond_FirstUseEver)
    ig.SetNextWindowSize((180.0f0, 0.0f0), ig.ImGuiCond_FirstUseEver)
    flags = ig.ImGuiWindowFlags_AlwaysAutoResize
    ig.Begin("Tools", C_NULL, flags)
    labels = ("Select", "Point", "Line", "Circle")
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
        changed = tool_button("Select", :select, tool, button_size)
        ig.TableNextColumn()
        changed = tool_button("Point", :point, tool, button_size) || changed
        ig.TableNextColumn()
        changed = tool_button("Line", :line, tool, button_size) || changed
        ig.TableNextColumn()
        changed = tool_button("Circle", :circle, tool, button_size) || changed
        ig.EndTable()
        if changed
            line_start[] = 0
            circle_center[] = 0
            dragging[] = 0
        end
    end
    ig.PopStyleVar()
    ig.End()
end

function draw_status_panel!(stats, report, solve_time_ms)
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
    ig.TextUnformatted(@sprintf("solve: %.2f ms", solve_time_ms))

    if report.conflicted && !isempty(report.entries)
        ig.Separator()
        ig.TextUnformatted("Top residuals")
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

function draw_selection_panel!(sketch, selected, residuals)
    vp = unsafe_load(ig.GetMainViewport())
    padding = 12.0f0
    ig.SetNextWindowPos((vp.Pos.x + vp.Size.x - padding, vp.Pos.y + padding),
                        ig.ImGuiCond_Always, (1.0f0, 0.0f0))
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

    if selected[] == 0
        ig.TextUnformatted("Selected: none")
    else
        x, y = point_xy(sketch, selected[])
        ig.TextUnformatted(@sprintf("Selected p%d", selected[]))
        ig.TextUnformatted(@sprintf("pos: (%.3f, %.3f)", x, y))
        for (idx, line) in enumerate(sketch.lines)
            if line_has_point(line, selected[])
                ig.TextUnformatted(@sprintf("line l%d: p%d-p%d", idx, line.p1, line.p2))
            end
        end
    end

    ig.Separator()
    ig.TextUnformatted("Constraints")
    if ig.BeginChild("constraints", (0.0f0, 0.0f0), true)
        for (idx, constraint) in enumerate(sketch.constraints)
            label = @sprintf("%d. %s", idx, constraint_label(constraint, sketch))
            res = @sprintf("r=%.3e", residuals[idx])
            line_start = ig.GetCursorPosX()
            ig.SetCursorPosX(line_start)
            ig.TextUnformatted(label)
            res_w = ig.CalcTextSize(res).x
            ig.SameLine()
            res_x = line_start + max_label_w + 12.0f0 + (max_res_w - res_w)
            ig.SetCursorPosX(res_x)
            ig.TextUnformatted(res)
        end
        ig.EndChild()
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
    push!(sketch, CADConstraints.Distance(p1, p2, d12 * 1.2))
    push!(sketch, CADConstraints.Distance(p2, p3, d23))
    push!(sketch, CADConstraints.Distance(p3, p4, d34))
    push!(sketch, CADConstraints.Distance(p3, p5, d35))
    t0 = time_ns()
    stats_ref = Ref(CADConstraints.solve!(sketch))
    solve_time_ms = Ref((time_ns() - t0) / 1e6)
    report_ref = Ref(CADConstraints.conflicts(sketch, stats_ref[]))
    residuals_ref = Ref(Float64[])
    constraint_residuals!(residuals_ref[], sketch)
    @info "Initial solve" iters=stats_ref[].iters status=stats_ref[].status cost=stats_ref[].cost
    for idx in 1:point_count(sketch)
        x, y = point_xy(sketch, idx)
        @info "Point" index=idx x y
    end
    selected = Ref(0)
    hovered = Ref(0)
    dragging = Ref(0)
    tool = Ref(:select)
    line_start = Ref(0)
    circle_center = Ref(0)
    center = Ref((0.0, 0.0))
    scale = Ref(80.0)

    ig.render(ctx; window_size=window_size, window_title=window_title) do
        draw_sketch!(sketch, selected, hovered, dragging, tool, line_start, circle_center, center, scale, stats_ref, report_ref, residuals_ref, solve_time_ms)
        draw_toolbar!(tool, line_start, circle_center, dragging)
        draw_status_panel!(stats_ref[], report_ref[], solve_time_ms[])
        draw_selection_panel!(sketch, selected, residuals_ref[])
    end
    return nothing
end

end # module CADSketchUI
