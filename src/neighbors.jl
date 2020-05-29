# nearest neighbor related functionality

abstract type NeighborParams end

# for finding approximate nearest neighbors
struct DescentNeighbors{M, K} <: NeighborParams
    n_neighbors::Int
    metric::M
    kwargs::K
end

# for precomputed distance matrix
struct PrecomputedNeighbors{M} <: NeighborParams
    n_neighbors::Int
    dists::M
end

# finding neighbors

# for each view, find the knns (fit)
function knn_search(data::NamedTuple{T}, knn_params::NamedTuple{T}) where T
    view_graphs = [knn_search(data[name], knn_params[name]) for name in keys(data)]
    return NamedTuple{T}(tuple(view_graphs...))
end

# NOTE: here, data will need to be the UMAP result struct
# for each view, find the knn graph for the query data
function knn_search(result, queries::NamedTuple{T}, knn_params::NamedTuple{T}) where T
    # if result::UMAPResult{DS, DT, C, V} where result.views::V, then
    # V will be a named tuple of UMAPViewResult here
    view_graphs = [knn_search(result.data[name],
                              queries[name],
                              knn_params[name],
                              result.views[name]) for name in keys(result.data)]
    return NamedTuple{T}(tuple(view_graphs...))
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
