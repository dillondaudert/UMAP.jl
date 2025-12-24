# nearest neighbor related functionality

# NEIGHBOR PARAMS
"""
Structs for parameterizing the knn search step of UMAP.

Subtyping `NeighborParams` allows for different methods of finding nearest neighbors.
The `knn_search` function will dispatch on the type of `knn_params` to
find the nearest neighbors in the data.
"""
abstract type NeighborParams end

"""
    DescentNeighbors(n_neighbors, metric, kwargs)

Parameters for finding approximate nearest neighbors using NearestNeighborDescent.
"""
struct DescentNeighbors{M, K} <: NeighborParams
    "The number of neighbors to consider as connected. The more neighbors, the more global structure is captured in the data."
    n_neighbors::Int
    "A distance function for comparing two points"
    metric::M
    "Keyword arguments to pass to NearestNeighborDescent.nndescent()"
    kwargs::K
end
DescentNeighbors(n_neighbors, metric) = DescentNeighbors(n_neighbors, metric, NamedTuple())

"""
    PrecomputedNeighbors(n_neighbors, dists_or_graphs)

Parameters for finding nearest neighbors from precomputed distances.
"""
struct PrecomputedNeighbors{M} <: NeighborParams
    n_neighbors::Int
    dists_or_graph::M
end


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

(FIT) Map `knn_search` over each view of the data and corresponding knn_params.
"""
function knn_search(data::NamedTuple{T}, knn_params::NamedTuple{T}) where T
    return map(knn_search, data, knn_params)
end


"""
    knn_search(data::NamedTuple{T}, queries::NamedTuple{T}, knn_params::NamedTuple{T}, result_knns_dists::NamedTuple{T}) -> NamedTuple{T}

(TRANSFORM) Map `knn_search` over each view of the data, queries, knn_params, and
result_knns_dists.
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
The (knns, dists) are used to reconstruct the original KNN graph.
"""
function knn_search(data, queries, knn_params::DescentNeighbors, (knns, dists))
    orig_knn_graph = HeapKNNGraph(data,
                                  knn_params.metric,
                                  knns,
                                  dists)
    return search(orig_knn_graph, queries, knn_params.n_neighbors; knn_params.kwargs...)
end

# get neighbors from precomputed KNNGraph
"""
    knn_search(_, knn_params::PrecomputedNeighbors{M}) where {M <: ApproximateKNNGraph}

Get neighbors from a precomputed KNNGraph.
This method is used when the KNN graph has been precomputed using NearestNeighborDescent
"""
function knn_search(_, knn_params::PrecomputedNeighbors{M}) where {M <: ApproximateKNNGraph}
    knn_graph = knn_params.dists_or_graph
    return knn_matrices(knn_graph)
end


"""
    knn_search(_, knn_params::PrecomputedNeighbors)

(FIT) Get neighbors from a precomputed distance matrix.
"""
function knn_search(_, knn_params::PrecomputedNeighbors)
    return _knn_from_dists(knn_params.dists_or_graph, knn_params.n_neighbors)
end
"""
    knn_search(_, __, knn_params::PrecomputedNeighbors, ___)

(TRANSFORM) Get neighbors from a precomputed distance matrix.
"""
function knn_search(_, __, knn_params::PrecomputedNeighbors, ___)
    return _knn_from_dists(knn_params.dists_or_graph, knn_params.n_neighbors; ignore_diagonal=false)
end


"""
    _knn_from_dists(dist_mat::AbstractMatrix{S}, k::Integer; ignore_diagonal=true) where {S <: Real}

Construct k-nearest neighbors and distances from a distance matrix.
If `ignore_diagonal` is true, the diagonal elements (which are 0) will
be ignored when determining the k-nearest neighbors. This is useful when
the distance matrix represents pairwise distances of the same set, where
the diagonal elements are the distances to themselves and will always be 0.

Returns a tuple of two matrices: the first contains the indices of the k-nearest neighbors,
and the second contains the corresponding distances to those neighbors.
"""
function _knn_from_dists(dist_mat::AbstractMatrix{S}, k::Integer; ignore_diagonal=true) where {S <: Real}
    # Ignore diagonal 0 elements (which will be smallest) when distance matrix represents pairwise distances of the same set
    # If dist_mat represents distances between two different sets, diagonal elements be nontrivial
    range = (1:k) .+ ignore_diagonal
    knns  = Array{Int,2}(undef,k,size(dist_mat,2))
    dists = Array{S,2}(undef,k,size(dist_mat,2))
    for i in axes(dist_mat, 2)
        knns[:,i]  = partialsortperm(dist_mat[ :, i], range)
        dists[:,i] = dist_mat[knns[:,i],i]
    end
    return knns, dists
end
