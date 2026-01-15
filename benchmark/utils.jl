# utilities for benchmarks
using Random
import NearestNeighborDescent as NND

function matrix_data(n_points, n_dims; seed=123456987)
    Random.seed!(seed)
    return randn(n_dims, n_points)
end

function vecvec_data(n_points, n_dims; seed=123456987)
    Random.seed!(seed)
    return [randn(n_dims) for _ in 1:n_points]
end

function knns_dists(n_points, in_dims, knn; metric=Euclidean())
    data = matrix_data(n_points, in_dims)
    
    knns, dists = NND.knn_matrices(NND.nndescent(data, knn, metric))
    return knns, dists
end
