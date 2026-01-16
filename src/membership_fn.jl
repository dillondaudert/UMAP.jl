

"""
    MembershipFnParams{T}(min_dist, spread)

A smooth approximation for the membership strength of a 1-simplex between two
points x, y, can be given by the following, with dissimilarity function `dist`,
and constants `a`, `b`:
`ϕ(x, y, dist, a, b) = (1 + a*(dist(x, y))^b)^(-1)`

The approximation parameters `a`, `b` (stored as Float32) are chosen by non-linear
least squares fitting of the following function ψ:

ψ(x, y, dist, min_dist, spread) = dist(x, y) ≤ min_dist ? 1 : exp(-(dist(x, y) - min_dist)/spread)
"""
mutable struct MembershipFnParams{T <: Real}
    "The minimum spacing of points in the target embedding"
    min_dist::T
    "The effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`."
    spread::T
    a::Float32
    b::Float32

    # first inner constructor fits the parameters
    function MembershipFnParams{T}(min_dist, spread) where {T <: Real}
        min_dist > 0 || throw(ArgumentError("min_dist must be greater than 0"))
        spread > 0 || throw(ArgumentError("spread must be greater than 0"))
        a, b = fit_ab(min_dist, spread)
        return new(min_dist, spread, a, b)
    end
    # inner constructor still checks min_dist, spread, but not a or b
    function MembershipFnParams{T}(min_dist, spread, a, b) where {T <: Real}
        min_dist > 0 || throw(ArgumentError("min_dist must be greater than 0"))
        spread > 0 || throw(ArgumentError("spread must be greater than 0"))
        return new(min_dist, spread, a, b)
    end
end
function MembershipFnParams(min_dist::T, spread::T) where {T <: Real}
    return MembershipFnParams{T}(min_dist, spread)
end
function MembershipFnParams(min_dist::T, spread::T, a::Float32, b::Float32) where {T <: Real}
    return MembershipFnParams{T}(min_dist, spread, a, b)
end
# autopromote
function MembershipFnParams(min_dist::Real, spread::Real)
    return MembershipFnParams(promote(min_dist, spread)...)
end
function MembershipFnParams(min_dist::Real, spread::Real, a::Real, b::Real)
    return MembershipFnParams(promote(min_dist, spread)..., convert.(Float32, (a, b))...)
end
# support passing a, b as nothing for convenience
function MembershipFnParams(min_dist::Real, spread::Real, ::Nothing, ::Nothing)
    return MembershipFnParams(min_dist, spread)
end


"""
    fit_ab(min_dist, spread) -> a, b

Find a smooth approximation to the membership function of points embedded in ℜᵈ.
This fits a smooth curve that approximates an exponential decay offset by `min_dist`,
returning the parameters `(a, b)`.
"""
function fit_ab(min_dist, spread)
    ψ(d) = d >= min_dist ? exp(-(d - min_dist)/spread) : 1.
    xs = LinRange(0., spread*3, 300)
    ys = map(ψ, xs)
    @. curve(x, p) = (1. + p[1]*x^(2*p[2]))^(-1)
    result = LsqFit.curve_fit(curve, xs, ys, [1., 1.], lower=[0., -Inf])
    a, b = result.param
    return a, b
end