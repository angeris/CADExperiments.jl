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
