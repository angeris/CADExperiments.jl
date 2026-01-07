# CADConstraints Development Plan (Draft)

## Goal
Build a fast, allocation‑minimal CAD constraint solver for interactive use. Start with points and lines only, then expand.

## Phase 0: Minimal Data Model
1) **Points are the only primitive**
   - All geometry is defined in terms of points `(x, y)` stored in a flat parameter vector.
2) **Shapes built from points**
   - A line is `(p1, p2)`; a circle is `(center, rim)`; future shapes follow this pattern.
3) **Constraints operate on shapes but resolve to points**
   - Each constraint references one or more shapes and expands to point-based residuals.
4) **Core structs**
   - `Sketch`: points, shapes, constraints, cached solver state.
   - `Shape`: lightweight references to points (no extra geometry state).
   - `Constraint`: residual count, sparsity pattern, callbacks (`r!`, `J!`).

## Phase 1: Basic Constraints (Points + Line Shapes)
5) **Point constraints**
   - Fixed point `(x0, y0)`.
   - Coincident points.
6) **Line constraints**
   - Horizontal/vertical line segment.
   - Point on line segment (collinearity).
   - Parallel lines (line1 ∥ line2).
7) **Constraint API**
   - `add_point!` for points, and `push!(sketch, Line(...))` for shapes.
   - `push!(sketch, FixedPoint(...))`, `Coincident(...)`, `Parallel(...)`,
     `Horizontal(...)`, `Vertical(...)` for constraints.
   - Avoid `push_constraint!` helpers; dispatch on `push!` is the only entrypoint.

## Phase 2: Solver Integration (SparseLNNS)
8) **Sparsity pattern construction**
   - Build `Jpat` once from constraint structure.
   - Ensure pattern includes structural zeros that can become nonzero.
9) **Workspace caching**
   - Allocate residual/Jacobian buffers once; reuse for repeated solves.
   - Store `SparseLNNS.State`/`Workspace` inside `Sketch`.
10) **Solve loop**
   - `solve!(sketch; options)` updates in place.
   - Allow “warm starts” when dragging points (reuse `state.x`).

## Phase 3: Real‑Time Updates
11) **Fast updates**
   - For drag edits: update `x` only, reuse constraint list and pattern.
   - For new constraints: rebuild `Jpat` and re‑initialize once.
12) **Dirty flags**
    - Track `structure_dirty` (pattern changed) vs `value_dirty` (x/parameter changed).
    - Do not rebuild `Problem` for value edits; reuse cached `Problem/State/Workspace`.

## Phase 4: Minimal Allocation Discipline
13) **No additional allocations**
    - `r!` and `J!` must write in place, with no extra allocations beyond structural changes
      (e.g., adding points/lines/constraints).
14) **Sparse writes**
    - Fill `nonzeros(J)` directly where possible.
15) **Microbenchmarks**
    - Bench drag update + solve; track allocations per solve.

## Phase 5: API for GUI (Future)
16) **Event‑oriented API**
    - `begin_drag!(point)`, `update_drag!(point, x, y)`, `end_drag!`.
17) **Status reporting**
    - Return convergence info and constraint residuals for UI feedback.

## Immediate Next Steps
- Implement `Sketch`, point/line storage, and the first 3–4 constraints.
- Build `Jpat` from constraints and integrate with `SparseLNNS`.
- Add tests for each constraint and a “drag” update path.

## Notes
- Treat zero‑length lines as coincident point constraints to avoid degenerate geometry.
- Keep each constraint’s row/column map fixed and fill `nonzeros(J)` in a stable order.
- Consider per‑constraint weights for soft constraints or UI “nudges”.
- Track `structure_dirty` vs `values_dirty` to avoid rebuilding patterns during drags.
- For interactive drags, allow low‑iteration solves and return residuals for UI feedback.
