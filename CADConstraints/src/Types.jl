abstract type Shape end
abstract type Constraint end

"""
    Line(p1, p2)

Line shape defined by two point indices.
"""
struct Line <: Shape
    p1::Int
    p2::Int
end

"""
    Circle(center, rim)

Circle shape defined by center and rim point indices.
"""
struct Circle <: Shape
    center::Int
    rim::Int
end

"""
    Arc(center, start, finish)

Arc shape defined by center, start, and end point indices.
"""
struct Arc <: Shape
    center::Int
    start::Int
    finish::Int
end

"""
    FixedPoint(p, x, y)

Fix point `p` at `(x, y)`.
"""
struct FixedPoint{T<:Real} <: Constraint
    p::Int
    x::T
    y::T
end

"""
    Coincident(p1, p2)

Constrain points `p1` and `p2` to coincide.
"""
struct Coincident <: Constraint
    p1::Int
    p2::Int
end

"""
    Horizontal(line)

Constrain a line to be horizontal (y1 == y2).
"""
struct Horizontal <: Constraint
    line::Int
end

"""
    Vertical(line)

Constrain a line to be vertical (x1 == x2).
"""
struct Vertical <: Constraint
    line::Int
end

"""
    Parallel(line1, line2)

Constrain two lines to be parallel.
"""
struct Parallel <: Constraint
    line1::Int
    line2::Int
end

"""
    Distance(p1, p2, d)

Constrain the distance between points `p1` and `p2` to `d`.
"""
struct Distance{T<:Real} <: Constraint
    p1::Int
    p2::Int
    d::T
end

"""
    Diameter(circle, d)

Constrain a circle's diameter to `d`.
"""
struct Diameter{T<:Real} <: Constraint
    circle::Int
    d::T
end

"""
    Normal(circle, line)

Constrain a line to pass through a circle center (normal direction).
"""
struct Normal <: Constraint
    circle::Int
    line::Int
end

"""
    CircleCoincident(circle, p)

Constrain point `p` to lie on the circle.
"""
struct CircleCoincident <: Constraint
    circle::Int
    p::Int
end
