# nearest neighbor related functionality

abstract type NeighborParams end

# for finding approximate nearest neighbors
struct DescentNeighbors{M, V, K} <: NeighborParams
    n_neighbors::Int
    metric::M
    cols::V
    kwargs::K
end

# for precomputed distance matrix
struct PrecomputedNeighbors{M} <: NeighborParams
    n_neighbors::Int
    dists::M
end

# utility function for getting particular views of the data
function get_data_view(data, knn_params::DescentNeighbors{M, ::Nothing}) where M
    # there is only one view: the entire dataset
    return data
end

function get_data_view(data, knn_params::DescentNeighbors)
    return Tables.getcolumn(data, knn_params.cols)
end

# finding neighbors

# broadcast over the knn parameters
function knn_search(data, knn_params)
    return knn_search.((data,), knn_params)
end

# NOTE: here, data will need to be the UMAP result struct
function knn_search(result, queries, knn_params)
    # if result::UMAPResult{DS, DT, C, V} where result.views::V, then
    # V <: UMAPViewResult (in other cases, it will be a vector of view results)
    return knn_search.((result.data,), (queries,), knn_params, result.views)
end

# find approximate neighbors

function knn_search(data, knn_params::DescentNeighbors)
    data_view = get_data_view(data, knn_params)
    knn_graph = nndescent(data_view, knn_params.n_neighbors, knn_params.metric; knn_params.kwargs...)
    return knn_matrices(knn_graph)
end

function knn_search(data, queries, knn_params::DescentNeighbors, view_result) where M
    data_view = get_data_view(data, knn_params)
    orig_knn_graph = HeapKNNGraph(data_view,
                                  knn_params.metric,
                                  result_view.knns,
                                  result_view.dists)
    queries_view = get_data_view(queries, knn_params)
    return search(orig_knn_graph, queries_view, knn_params.n_neighbors; knn_params.kwargs...)
end

# get neighbors from precomputed distance matrix

function knn_search(data, knn_params::PrecomputedNeighbors)
    return _knn_from_dists(knn_params.dists, knn_params.n_neighbors)
end

function knn_search(data, queries, knn_params::PrecomputedNeighbors, view_result)
    return _knn_from_dists(knn_params.dists, knn_params.n_neighbors; ignore_diagonal=true)
end
