struct Precomputed end

"""
    knn_search(dist_mat, k, ::Precomputed) -> knns, dists

Find the `k` nearest neighbors of each point in a precomputed distance 
matrix.
"""
function knn_search(dists, k, ::Precomputed)
    return _knn_from_dists(dists, k)
end

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
                    ::Val{:pairwise}) where {S <: AbstractFloat}
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
                    ::Val{:approximate}) where {S <: AbstractFloat}
    knngraph = DescentGraph(X, k, metric)
    return knngraph.indices, knngraph.distances    
end
    
function _knn_from_dists(dist_mat::AbstractMatrix{S}, k) where {S <: AbstractFloat}
    knns_ = [partialsortperm(view(dist_mat, :, i), 2:k+1) for i in 1:size(dist_mat, 1)]
    dists_ = [dist_mat[:, i][knns_[i]] for i in eachindex(knns_)]
    knns = hcat(knns_...)::Matrix{Int}
    dists = hcat(dists_...)::Matrix{S}
    return knns, dists
end

    
# combine local fuzzy simplicial sets 
@inline function combine_fuzzy_sets(fs_set::AbstractMatrix{T}, 
                            set_op_ratio::T) where {T <: AbstractFloat}
    return set_op_ratio .* fuzzy_set_union(fs_set) .+ 
           (one(T) - set_op_ratio) .* fuzzy_set_intersection(fs_set)
end

@inline function fuzzy_set_union(fs_set::AbstractMatrix)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

@inline function fuzzy_set_intersection(fs_set::AbstractMatrix)
    return fs_set .* fs_set'
end