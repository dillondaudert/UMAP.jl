

"""
    MembershipFnParams{T}(min_dist, spread, a, b)
"""
mutable struct MembershipFnParams{T <: Real}
    "The minimum spacing of points in the target embedding"
    min_dist::T
    "The effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`."
    spread::T
    a::T
    b::T
    function MembershipFnParams{T}(min_dist, spread, a, b) where {T <: Real}
        min_dist > 0 || throw(ArgumentError("min_dist must be greater than 0"))
        spread > 0 || throw(ArgumentError("spread must be greater than 0"))
        return new(min_dist, spread, a, b)
    end
end
function MembershipFnParams(min_dist::T, spread::T, a::T, b::T) where {T <: Real}
    return MembershipFnParams{T}(min_dist, spread, a, b)
end
# autopromote
function MembershipFnParams(min_dist::Real, spread::Real, a::Real, b::Real)
    return MembershipFnParams(promote(min_dist, spread, a, b)...)
end
# calculate a, b with binary search
function MembershipFnParams(min_dist::Real, spread::Real, ::Nothing, ::Nothing)
    a, b = fit_ab(min_dist, spread)
    return MembershipFnParams(min_dist, spread, a, b)
end
function MembershipFnParams(min_dist::Real, spread::Real)
    return MembershipFnParams(min_dist, spread, nothing, nothing)
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
    result = curve_fit(curve, xs, ys, [1., 1.], lower=[0., -Inf])
    a, b = result.param
    return a, b
end