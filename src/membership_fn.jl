

@inline fit_ab(_, __, a, b) = a, b

"""
    fit_ab(min_dist, spread, _a, _b) -> a, b

Find a smooth approximation to the membership function of points embedded in ℜᵈ.
This fits a smooth curve that approximates an exponential decay offset by `min_dist`,
returning the parameters `(a, b)`.
"""
function fit_ab(min_dist, spread, ::Nothing, ::Nothing)
    ψ(d) = d >= min_dist ? exp(-(d - min_dist)/spread) : 1.
    xs = LinRange(0., spread*3, 300)
    ys = map(ψ, xs)
    @. curve(x, p) = (1. + p[1]*x^(2*p[2]))^(-1)
    result = curve_fit(curve, xs, ys, [1., 1.], lower=[0., -Inf])
    a, b = result.param
    return a, b
end

"""
    fit_membership_fn(tgt_params::TargetParams)

Find a smooth approximation for the membership strength of a 1-simplex between two
points x, y. Returns a binary function `f(x, y)` as well as the gradient functions
`∇log(f(x, y))` and `∇log(1 - f(x, y))`.
"""
function fit_membership_fn end

function fit_membership_fn(tgt_params::TargetParams)
    a, b = fit_ab(tgt_params.min_dist, 
                  tgt_params.spread, 
                  tgt_params.a, 
                  tgt_params.b)
    fn = (x, y) -> 1 / (1 + a * tgt_params.metric(x, y)^b)
    ∇logfn = pos_grad(fn, tgt_params)
    ∇log1_fn = neg_grad(fn, tgt_params)
    return fn, ∇logfn, ∇log1_fn
end

function pos_grad(fn, tgt_params::TargetParams{_EuclideanManifold, SqEuclidean})
    function ∇logfn(x, y)
        sdist = SqEuclidean()(x, y)
        if 

