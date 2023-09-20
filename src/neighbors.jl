# nearest neighbor related functionality

# finding neighbors
"""
    knn_search(data, knn_params)
    knn_search(data, queries, knn_params, results)

Find the (potentially approximate) k-nearest neighbors for each point in the
dataset, according to knn_params. The data may consist of one or more 'views',
passed in either directly (for a single view only) or as a NamedTuple, with
corresponding knn_params (either an instance of <: NeighborParams or a NamedTuple
of such parameters).

For each (data_view, knn_params) pair, two KxN matrices are returned (knns, dists),
holding the indices of the k-nearest neighbors and distances to those neighbors
for each point in the dataset. When knn_params isa DescentNeighbors, these
matrices are computed using NearestNeighborDescent. When knn_params isa
PrecomputedNeighbors, the knn matrices are created from user-provided distances.

If data and knn_params are NamedTuples, the returned knn_matrices will also be
in a NamedTuple with the same keys as the inputs.
"""
function knn_search end

# for each view, find the knns (fit)
"""
    knn_search(data::NamedTuple{T}, knn_params::NamedTuple{T}) -> NamedTuple{T}
"""
function knn_search(data::NamedTuple{T}, knn_params::NamedTuple{T}) where T
    return map(knn_search, data, knn_params)
end


"""
    knn_search(data::NamedTuple{T}, queries::NamedTuple{T}, knn_params::NamedTuple{T}, result_knns_dists::NamedTuple{T}) -> NamedTuple{T}
"""
function knn_search(data::NamedTuple{T}, queries::NamedTuple{T}, knn_params::NamedTuple{T}, result_knns_dists::NamedTuple{T}) where T
    return map(knn_search, data, queries, knn_params, result_knns_dists)
end

# find approximate neighbors
"""
    knn_search(data, knn_params::DescentNeighbors) -> (knns, dists)

Find approximate nearest neighbors using nndescent.
"""
function knn_search(data, knn_params::DescentNeighbors)
    knn_graph = nndescent(data, knn_params.n_neighbors, knn_params.metric; knn_params.kwargs...)
    return knn_matrices(knn_graph)
end

"""
    knn_search(data, queries, knn_params::DescentNeighbors, result_knns_dists) -> (knns, dists)

Search for approximate nearest neighbors of queries in data using nndescent.
"""
function knn_search(data, queries, knn_params::DescentNeighbors, (knns, dists))
    orig_knn_graph = HeapKNNGraph(data,
                                  knn_params.metric,
                                  knns,
                                  dists)
    return search(orig_knn_graph, queries, knn_params.n_neighbors; knn_params.kwargs...)
end

# get neighbors from precomputed KNNGraph
function knn_search(data, knn_params::PrecomputedNeighbors{M}) where {M <: ApproximateKNNGraph}
    knn_graph = knn_params.dists_or_graph
    return knn_matrices(knn_graph)
end

# get neighbors from precomputed distance matrix
# fit
function knn_search(data, knn_params::PrecomputedNeighbors)
    return _knn_from_dists(knn_params.dists_or_graph, knn_params.n_neighbors)
end
# transform
function knn_search(data, queries, knn_params::PrecomputedNeighbors, knns_dists)
    return _knn_from_dists(knn_params.dists_or_graph, knn_params.n_neighbors; ignore_diagonal=false)
end

function _knn_from_dists(dist_mat::AbstractMatrix{S}, k::Integer; ignore_diagonal=true) where {S <: Real}
    # Ignore diagonal 0 elements (which will be smallest) when distance matrix represents pairwise distances of the same set
    # If dist_mat represents distances between two different sets, diagonal elements be nontrivial
    range = (1:k) .+ ignore_diagonal
    knns  = Array{Int,2}(undef,k,size(dist_mat,2))
    dists = Array{S,2}(undef,k,size(dist_mat,2))
    for i ∈ 1:size(dist_mat, 2)
        knns[:,i]  = partialsortperm(dist_mat[ :, i], range)
        dists[:,i] = dist_mat[knns[:,i],i]
    end
    return knns, dists
end
