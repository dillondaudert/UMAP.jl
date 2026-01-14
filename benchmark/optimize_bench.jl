# Benchmarks for embedding optimization

module OptimizeBench

using BenchmarkTools
using UMAP
using Distances
import NearestNeighborDescent as NND

include("utils.jl")

const N_POINTS = [1_000, 10_000]
const IN_DIMS = 10  # fixed for optimization benchmarks
const OUT_DIMS = [2, 10]
const KNN = 15
const N_EPOCHS = [50, 100]
const NEG_SAMPLE_RATES = [5]  # default value

optimize_suite = BenchmarkGroup(["optimization"])
optimize_suite["runtime"] = BenchmarkGroup()

"""
    make_optimization_inputs(n_points, in_dims, out_dims, k)

Create all inputs needed for benchmarking optimize_embedding!:
- umap_graph: the sparse UMAP graph
- embedding: initialized embedding (copy for each benchmark run)
- tgt_params: target parameters
"""
function make_optimization_inputs(n_points, in_dims, out_dims, k)
    # Create UMAP graph
    knns, dists = knns_dists(n_points, in_dims, k)
    knn_params = UMAP.DescentNeighbors(k, Euclidean())
    src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)
    graph = UMAP.fuzzy_simplicial_set((knns, dists), n_points, knn_params, src_params, true)

    # Create target params
    manifold = UMAP._EuclideanManifold(out_dims)
    memb_params = UMAP.MembershipFnParams(0.1, 1.0)  # min_dist=0.1, spread=1.0
    tgt_params = UMAP.TargetParams(manifold, SqEuclidean(), UMAP.SpectralInitialization(), memb_params)

    # Initialize embedding
    embedding = UMAP.initialize_embedding(graph, manifold, UMAP.UniformInitialization())

    return graph, embedding, tgt_params
end

# Optimization benchmarks varying n_points, out_dims, n_epochs
for (n_points, out_dims, n_epochs) in Iterators.product(N_POINTS, OUT_DIMS, N_EPOCHS)
    graph, embedding, tgt_params = make_optimization_inputs(n_points, IN_DIMS, out_dims, KNN)
    opt_params = UMAP.OptimizationParams(n_epochs, 1.0, 1.0, 5)

    key = "$(n_points)x$(out_dims)_$(n_epochs)epochs"

    # Runtime benchmarks - need to copy embedding since optimize_embedding! is mutating
    optimize_suite["runtime"][key] = @benchmarkable(
        UMAP.optimize_embedding!(emb, $graph, $tgt_params, $opt_params),
        setup=(GC.gc(); emb = deepcopy($embedding)),
        evals=1,
        samples=5,
    )
end

# Additional benchmark: varying neg_sample_rate (with fixed n_epochs)
for (n_points, neg_rate) in Iterators.product([10_000], [1, 5, 10])
    graph, embedding, tgt_params = make_optimization_inputs(n_points, IN_DIMS, 2, KNN)
    opt_params = UMAP.OptimizationParams(50, 1.0, 1.0, neg_rate)

    key = "$(n_points)_negrate$(neg_rate)"

    optimize_suite["runtime"][key] = @benchmarkable(
        UMAP.optimize_embedding!(emb, $graph, $tgt_params, $opt_params),
        setup=(GC.gc(); emb = deepcopy($embedding)),
        evals=1,
        samples=5,
    )
end

end # module
