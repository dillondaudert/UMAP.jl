# nearest neighbor related functionality

# finding neighbors

# for each view, find the knns (fit)
function knn_search(data::NamedTuple{T}, knn_params::NamedTuple{T}) where T
    return map(knn_search, data, knn_params)
end

# NOTE: here, data will need to be the UMAP result struct
# for each view, find the knn graph for the query data
# TODO: transform methods once result struct is better defined
function knn_search(result, queries::NamedTuple{T}, knn_params::NamedTuple{T}) where T
    # if result::UMAPResult{DS, DT, C, V} where result.views::V, then
    # V will be a named tuple of UMAPViewResult here
    return map(knn_search, result.data, queries, knn_params, result.views)
end

# find approximate neighbors

function knn_search(data, knn_params::DescentNeighbors)
    knn_graph = nndescent(data, knn_params.n_neighbors, knn_params.metric; knn_params.kwargs...)
    return knn_matrices(knn_graph)
end

function knn_search(data, queries, knn_params::DescentNeighbors, view_result)
    orig_knn_graph = HeapKNNGraph(data_view,
                                  knn_params.metric,
                                  result_view.knns,
                                  result_view.dists)
    return search(orig_knn_graph, queries_view, knn_params.n_neighbors; knn_params.kwargs...)
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
