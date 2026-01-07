module CADSketchUI

import CImGui as ig
import GLFW
import ModernGL

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

@inline function color_u32(r::Float32, g::Float32, b::Float32, a::Float32 = 1.0f0)
    return ig.GetColorU32(ig.ImVec4(r, g, b, a))
end

function draw_sketch!(points, lines, selected, hovered, center, scale)
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
    for (idx, p) in enumerate(points)
        sp = world_to_screen(p, origin, center[], scale_f)
        if dist2(sp, mouse) <= hit_radius * hit_radius
            hovered[] = idx
            break
        end
    end

    if is_hovered && ig.IsMouseClicked(0)
        selected[] = hovered[]
    end

    for (i, j) in lines
        p1 = world_to_screen(points[i], origin, center[], scale_f)
        p2 = world_to_screen(points[j], origin, center[], scale_f)
        ig.AddLine(draw_list, p1, p2, color_u32(0.55f0, 0.65f0, 0.75f0), 2.0f0)
    end

    for (idx, p) in enumerate(points)
        sp = world_to_screen(p, origin, center[], scale_f)
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

function draw_toolbar!()
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
        if ig.Button("Select", button_size)
            @info "Tool: Select"
        end
        ig.TableNextColumn()
        if ig.Button("Point", button_size)
            @info "Tool: Point"
        end
        ig.TableNextColumn()
        if ig.Button("Line", button_size)
            @info "Tool: Line"
        end
        ig.TableNextColumn()
        if ig.Button("Circle", button_size)
            @info "Tool: Circle"
        end
        ig.EndTable()
    end
    ig.PopStyleVar()
    ig.End()
end

"""
    run(; window_size=(640, 480), window_title="CADSketchUI")

Open a minimal sketch window with a canvas and a few placeholder points/lines.
This is the first UI checkpoint (canvas + selection + pan/zoom).
"""
function run(; window_size=(640, 480), window_title="CADSketchUI")
    ig.set_backend(:GlfwOpenGL3)
    ctx = ig.CreateContext()

    points = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.2)]
    lines = [(1, 2), (2, 3)]
    selected = Ref(0)
    hovered = Ref(0)
    center = Ref((0.0, 0.0))
    scale = Ref(80.0)

    ig.render(ctx; window_size=window_size, window_title=window_title) do
        draw_sketch!(points, lines, selected, hovered, center, scale)
        draw_toolbar!()
    end
    return nothing
end

end # module CADSketchUI
