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


"""
    knn_search(X::AbstractMatrix, k::Integer, metric)                                -> knns, dists
    knn_search(X::AbstractMatrix, Q::AbstractMatrix, k::Integer, metric::SemiMetric,
                    knns::AbstractMatrix{<:Integer}, dists::AbstractMatrix{<:Real})  -> knns, dists,

Find the `k` nearest neighbors of each point.

`metric` may be of type:
- ::Symbol - `knn_search` is dispatched to one of the following based on the evaluation of `metric`:
- ::Val(:precomputed) - computes neighbors from `X` treated as a precomputed distance matrix.
- ::SemiMetric - computes neighbors from `X` treated as samples, using the given metric.

Given a matrix `X` and a matrix `Q`, use the given metric to compute the `k` nearest neighbors out of the
columns of `X` from the queries (columns in `Q`). 
If the matrices are large, compute the approximate nearest neighbors graph of `X` and use this to search 
for approximate nearest neighbors of `Q`. If provided `knns` and `dists` are nonempty, use these to reconstruct 
the approximate nearest neighbors graph; otherwise, reconstruct it from scratch.
If the matrices are small, search for exact nearest neighbors of `Q` by computing all pairwise distances with `X`.

# Returns
- `knns`: `knns[j, i]` is the index of node i's jth nearest neighbor.
- `dists`: `dists[j, i]` is the distance of node i's jth nearest neighbor.
"""
function knn_search end

knn_search(X::AbstractMatrix, k::Integer, metric::Symbol) = knn_search(X, k, Val(metric))

# treat given matrix `X` as distance matrix
knn_search(X::AbstractMatrix, k::Integer, ::Val{:precomputed}) = _knn_from_dists(X, k)

function knn_search(X::AbstractMatrix,
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
                    k::Integer,
                    metric,
                    ::Val{:pairwise}) where {S <: Real}
    num_points = size(X, 2)
    if S <: AbstractFloat
        dist_mat = Array{S}(undef, num_points, num_points)
    else
        dist_mat = Array{Float64}(undef, num_points, num_points)
    end
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

function knn_search(X::AbstractMatrix, 
                    Q::AbstractMatrix,
                    k::Integer,
                    metric::SemiMetric,
                    knns::AbstractMatrix{<:Integer},
                    dists::AbstractMatrix{<:Real})
    
    if size(X, 2) < 4096
        return _knn_from_dists(pairwise(metric, X, Q, dims=2), k, ignore_diagonal=false)
    end

    if isempty(knns) || isempty(dists) || size(knns) != size(dists) || size(knns, 1) < k
        knngraph = nndescent(X, k, metric)
    else
        knngraph = HeapKNNGraph(collect(eachcol(X)), metric, knns, dists)
    end
    return search(knngraph, collect(eachcol(Q)), k)
end


function _knn_from_dists(dist_mat::AbstractMatrix{S}, k::Integer; ignore_diagonal=true) where {S <: Real}
    # Ignore diagonal 0 elements (which will be smallest) when distance matrix represents pairwise distances of the same set
    # If dist_mat represents distances between two different sets, diagonal elements be nontrivial
    range = (1:k) .+ ignore_diagonal
    knns_ = [partialsortperm(view(dist_mat, :, i), range) for i in 1:size(dist_mat, 2)]
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
