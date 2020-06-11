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


# TODO: transform methods once result struct is better defined
"""
    knn_search(data::NamedTuple{T}, queries::NamedTuple{T}, knn_params::NamedTuple{T}, view_result::NamedTuple{T}) -> NamedTuple{T}
"""
function knn_search(data::NamedTuple{T}, queries::NamedTuple{T}, knn_params::NamedTuple{T}, view_result::NamedTuple{T}) where T
    # if result::UMAPResult{DS, DT, C, V} where result.views::V, then
    # V will be a named tuple of UMAPViewResult here
    return map(knn_search, data, queries, knn_params, view_result)
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
    knn_search(data, queries, knn_params::DescentNeighbors, result) -> (knns, dists)

Search for approximate nearest neighbors of queries in data using nndescent.
"""
function knn_search(data, queries, knn_params::DescentNeighbors, view_result)
    orig_knn_graph = HeapKNNGraph(data,
                                  knn_params.metric,
                                  view_result.knns,
                                  view_result.dists)
    return search(orig_knn_graph, queries, knn_params.n_neighbors; knn_params.kwargs...)
end

# get neighbors from precomputed distance matrix

function knn_search(data, knn_params::PrecomputedNeighbors)
    return _knn_from_dists(knn_params.dists, knn_params.n_neighbors)
end

function knn_search(data, queries, knn_params::PrecomputedNeighbors, view_result)
    return _knn_from_dists(knn_params.dists, knn_params.n_neighbors; ignore_diagonal=true)
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
