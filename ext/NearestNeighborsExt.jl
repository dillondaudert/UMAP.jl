# Package extension providing exact knn search for ExactNeighbors params
# via NearestNeighbors.jl trees.
module NearestNeighborsExt

import UMAP
import UMAP: ExactNeighbors
import NearestNeighbors as NN
import Distances

# metrics with axis-aligned structure that KDTree supports; fall back to BallTree otherwise
const KDTreeMetric = Union{Distances.Euclidean, Distances.Chebyshev, Distances.Cityblock, Distances.Minkowski}

_build_tree(data, knn_params::ExactNeighbors{<:KDTreeMetric}) = NN.KDTree(data, knn_params.metric; knn_params.kwargs...)
_build_tree(data, knn_params::ExactNeighbors) = NN.BallTree(data, knn_params.metric; knn_params.kwargs...)

# NearestNeighbors requires statically sized points (matrix columns or SVectors);
# convert generic vectors-of-vectors to a matrix
_pointmatrix(data::AbstractMatrix) = data
_pointmatrix(data::AbstractVector) = reduce(hcat, data)

# FIT: each point's own entry is excluded from its neighbors, mirroring
# _knn_from_dists(...; ignore_diagonal=true)
function UMAP._exact_knn(data, knn_params::ExactNeighbors)
    k = knn_params.n_neighbors
    points = _pointmatrix(data)
    tree = _build_tree(points, knn_params)
    knns_list, dists_list = NN.knn(tree, points, k + 1, true)
    n_points = length(knns_list)
    knns = Matrix{Int}(undef, k, n_points)
    dists = Matrix{eltype(eltype(dists_list))}(undef, k, n_points)
    for i in 1:n_points
        # drop point i from its own neighbor list; if absent (duplicate points),
        # drop the nearest neighbor (also distance 0) instead
        self_idx = something(findfirst(==(i), knns_list[i]), 1)
        keep = deleteat!(collect(1:(k + 1)), self_idx)
        knns[:, i] = knns_list[i][keep]
        dists[:, i] = dists_list[i][keep]
    end
    return knns, dists
end

# TRANSFORM: queries are distinct from data, so no self-exclusion
function UMAP._exact_knn(data, queries, knn_params::ExactNeighbors)
    tree = _build_tree(_pointmatrix(data), knn_params)
    knns_list, dists_list = NN.knn(tree, _pointmatrix(queries), knn_params.n_neighbors, true)
    return reduce(hcat, knns_list), reduce(hcat, dists_list)
end

end # module
