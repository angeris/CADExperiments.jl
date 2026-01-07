module CADConstraints

using SparseArrays
using SparseLNNS

import Base: push!

export Shape, Constraint, Line, Circle, Arc
export FixedPoint, Coincident, CircleCoincident, Horizontal, Vertical, Parallel, Distance, Diameter, Normal
export Sketch, add_point!, set_point!, build_problem!, solve!
export residual_norm, has_conflict, conflicts, ConflictEntry, ConflictReport

include("Types.jl")
include("Sketch.jl")
include("Constraints.jl")
include("Conflicts.jl")

end # module CADConstraints
