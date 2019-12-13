#=
Utilities used by UMAP.jl
=#


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

knn_search(dist_mat, k, metric::Symbol) = knn_search(dist_mat, k, Val(metric))

"""
    knn_search(dist_mat, k, :precomputed) -> knns, dists

Find the `k` nearest neighbors of each point in a precomputed distance
matrix.
"""
knn_search(dist_mat, k, ::Val{:precomputed}) = _knn_from_dists(dist_mat, k)

"""
    knn_search(X, k, metric) -> knns, dists

Find the `k` nearest neighbors of each point in `X` by `metric`.
"""
function knn_search(X,
                    k,
                    metric::SemiMetric)
    if size(X, 2) < 4096
        return knn_search(X, k, metric, Val(:pairwise))
    else
        return knn_search(X, k, metric, Val(:approximate))
    end
end

# compute all pairwise distances
# return the nearest k to each point v, other than v itself
function knn_search(X::AbstractMatrix{S},
                    k,
                    metric,
                    ::Val{:pairwise}) where {S <: Real}
    num_points = size(X, 2)
    dist_mat = Array{S}(undef, num_points, num_points)
    pairwise!(dist_mat, metric, X, dims=2)
    # all_dists is symmetric distance matrix
    return _knn_from_dists(dist_mat, k)
end

# find the approximate k nearest neighbors using NNDescent
function knn_search(X::AbstractMatrix{S},
                    k,
                    metric,
                    ::Val{:approximate}) where {S <: Real}
    knngraph = nndescent(X, k, metric)
    return knn_matrices(knngraph)
end

function _knn_from_dists(dist_mat::AbstractMatrix{S}, k) where {S <: Real}
    knns_ = [partialsortperm(view(dist_mat, :, i), 2:k+1) for i in 1:size(dist_mat, 1)]
    dists_ = [dist_mat[:, i][knns_[i]] for i in eachindex(knns_)]
    knns = hcat(knns_...)::Matrix{Int}
    dists = hcat(dists_...)::Matrix{S}
    return knns, dists
end


# combine local fuzzy simplicial sets
@inline function combine_fuzzy_sets(fs_set::AbstractMatrix{T},
                                    set_op_ratio) where {T}
    return set_op_ratio .* fuzzy_set_union(fs_set) .+
           (one(T) - set_op_ratio) .* fuzzy_set_intersection(fs_set)
end

@inline function fuzzy_set_union(fs_set::AbstractMatrix)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

@inline function fuzzy_set_intersection(fs_set::AbstractMatrix)
    return fs_set .* fs_set'
end
