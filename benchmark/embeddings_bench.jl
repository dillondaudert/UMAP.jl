# Benchmarks for embedding initialization

module EmbeddingsBench

using BenchmarkTools
using UMAP
using Distances
import NearestNeighborDescent as NND

include("utils.jl")

const N_POINTS = [1_000, 10_000]
const IN_DIMS = [10, 50]
const OUT_DIMS = [2, 10]
const KNN = 15

embeddings_suite = BenchmarkGroup(["embeddings"])
embeddings_suite["spectral"] = BenchmarkGroup(["runtime"])
embeddings_suite["uniform"] = BenchmarkGroup(["runtime"])

"""
    make_umap_graph(n_points, in_dims, k)

Create a UMAP graph (sparse matrix) for benchmarking embedding initialization.
"""
function make_umap_graph(n_points, in_dims, k)
    knns, dists = knns_dists(n_points, in_dims, k)
    knn_params = UMAP.DescentNeighbors(k, Euclidean())
    src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)
    graph = UMAP.fuzzy_simplicial_set((knns, dists), n_points, knn_params, src_params, true)
    return graph
end

# Spectral initialization benchmarks
for (n_points, in_dims, out_dims) in Iterators.product(N_POINTS, IN_DIMS, OUT_DIMS)
    graph = make_umap_graph(n_points, in_dims, KNN)
    manifold = UMAP._EuclideanManifold(out_dims)
    init = UMAP.SpectralInitialization()

    embeddings_suite["spectral"]["$(n_points)x$(in_dims)->$(out_dims)"] = @benchmarkable(
        UMAP.initialize_embedding($graph, $manifold, $init),
        setup=(GC.gc()),
        evals=1,
        samples=10,
    )
end

# Uniform initialization benchmarks
for (n_points, in_dims, out_dims) in Iterators.product(N_POINTS, IN_DIMS, OUT_DIMS)
    graph = make_umap_graph(n_points, in_dims, KNN)
    manifold = UMAP._EuclideanManifold(out_dims)
    init = UMAP.UniformInitialization()

    embeddings_suite["uniform"]["$(n_points)x$(in_dims)->$(out_dims)"] = @benchmarkable(
        UMAP.initialize_embedding($graph, $manifold, $init),
        setup=(GC.gc()),
        evals=2,
        samples=20,
    )
end

end # module
