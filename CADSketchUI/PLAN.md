# CADSketchUI Plan (Draft)

## Goal
Build a minimal, fast, interactive 2D sketching UI that drives `CADConstraints` in real time. Keep the UI simple and responsive first; expand tooling later.

## Package & Stack Choice (Decision)
Start with **CImGui.jl** for immediate‑mode UI and input handling:
- Use `CImGui` for docking, tool panels, and input events.
- Use `CImGui.GLFWBackend` + `CImGui.OpenGLBackend` for a desktop window.
- Render the sketch into an OpenGL canvas area inside the ImGui window.

Immediate‑mode fits interactive editing well (hit‑testing, dragging, tools). We can
add a dedicated renderer later if needed, but the first pass should stay inside ImGui.

## Folder Layout
- `CADSketchUI/` (new package folder, separate from `CADConstraints` and `SparseLNNS`)
- `CADSketchUI/PLAN.md` (this doc)
- Eventually: `CADSketchUI/src/`, `CADSketchUI/test/`

## UI Architecture (High Level)
1) **SketchModel**
   - Wraps a `CADConstraints.Sketch` and exposes point/shape handles.
   - Owns `solve!` options, logging toggles, and dirty flags.
2) **Render Layer**
   - Use `ImDrawList` to draw points/lines/circles directly in the ImGui window.
   - No custom OpenGL pipeline in v1; keep rendering inside ImGui.
   - Maintain selection/hover state (indices + styles).
3) **Interaction Layer**
   - Hit test for points/segments; begin drag → update point → solve → redraw.
   - Basic tools: add point, add line, add circle; constraint tools wired to selections.
4) **Solver Integration**
   - Warm‑start from current state, avoid rebuilding `Problem` on drags.
   - Throttle solves (e.g., solve every N ms during drag; always on mouse‑up).

## Performance & Allocation Discipline
- UI is allowed to allocate, but solving should not allocate beyond structural edits.
- Keep geometry buffers stable; update `Observables` with in‑place arrays.
- Avoid per‑frame recomputation of constraint structure.

## Simplest Viable First Pass (Detailed)
1) **Window + Canvas**
   - Create a single docked window with a toolbar at top and a canvas region.
   - Use the canvas size from `CImGui.GetContentRegionAvail()`; draw into that rect.
2) **Coordinate System**
   - Maintain a 2D camera: `origin_px` (screen offset) and `scale` (px per unit).
   - Screen ↔ world conversion is just `world = (screen - origin_px) / scale`.
3) **Rendering**
   - Points: small filled circles (3–5 px radius).
   - Lines: thick line segments (1–2 px).
   - Circles: polyline approximation (e.g., 64 segments; adjustable).
4) **Hit‑Testing**
   - Points: screen‑space distance to cursor <= selection radius.
   - Lines: distance to segment in screen space.
   - Circles: distance to center vs radius (screen space).
5) **Interaction**
   - On mouse‑down: choose nearest selectable entity.
   - On drag: update point positions in the `Sketch`, call `solve!`, redraw.
   - On mouse‑up: finalize; emit a solve summary in a small HUD panel.
6) **Minimal UI Panels**
   - Toolbar: select tool, add point, add line, add circle.
   - Inspector: show selected entity and constraint residuals.
   - Diagnostics: iteration count, residual norm, time/alloc (from solver).

## Testing & Benchmarks (Later)
- Smoke test: create a sketch, add constraints, drag point, verify convergence.
- Allocation check: repeated drag + solve should not allocate.
- “Wiggle” benchmark for end‑to‑end UI responsiveness.

## Milestones
1) Minimal canvas with points/lines rendered, click‑to‑select.
2) Drag point + warm‑started solve loop.
3) Add constraint tools and simple HUD/overlay for diagnostics.
