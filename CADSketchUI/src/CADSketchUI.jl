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

@inline function world_to_screen(p, origin::ig.ImVec2, scale::Cfloat)
    return ig.ImVec2(origin.x + Cfloat(p[1]) * scale, origin.y - Cfloat(p[2]) * scale)
end

@inline function color_u32(r::Float32, g::Float32, b::Float32, a::Float32 = 1.0f0)
    return ig.GetColorU32(ig.ImVec4(r, g, b, a))
end

function draw_sketch!(points, lines, selected, hovered, scale)
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

    scale_f = Cfloat(scale)
    origin = ig.ImVec2(canvas_pos.x + canvas_size.x * 0.5f0, canvas_pos.y + canvas_size.y * 0.5f0)
    mouse = ig.GetMousePos()

    hovered[] = 0
    hit_radius = 8.0f0
    for (idx, p) in enumerate(points)
        sp = world_to_screen(p, origin, scale_f)
        if dist2(sp, mouse) <= hit_radius * hit_radius
            hovered[] = idx
            break
        end
    end

    if is_hovered && ig.IsMouseClicked(0)
        selected[] = hovered[]
    end

    for (i, j) in lines
        p1 = world_to_screen(points[i], origin, scale_f)
        p2 = world_to_screen(points[j], origin, scale_f)
        ig.AddLine(draw_list, p1, p2, color_u32(0.55f0, 0.65f0, 0.75f0), 2.0f0)
    end

    for (idx, p) in enumerate(points)
        sp = world_to_screen(p, origin, scale_f)
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

"""
    run(; window_size=(640, 480), window_title="CADSketchUI")

Open a minimal sketch window with a canvas and a few placeholder points/lines.
This is the first UI checkpoint (canvas + selection).
"""
function run(; window_size=(640, 480), window_title="CADSketchUI")
    ig.set_backend(:GlfwOpenGL3)
    ctx = ig.CreateContext()

    points = [(-1.0, 0.0), (1.0, 0.0), (0.0, 1.2)]
    lines = [(1, 2), (2, 3)]
    selected = Ref(0)
    hovered = Ref(0)
    scale = 80.0f0

    ig.render(ctx; window_size=window_size, window_title=window_title) do
        draw_sketch!(points, lines, selected, hovered, scale)
    end
    return nothing
end

end # module CADSketchUI
