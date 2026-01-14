# Benchmark suite for UMAP.jl
#
# This suite covers the major UMAP algorithm stages:
# - Fuzzy simplicial set construction (simplicial_sets_bench.jl)
# - Embedding initialization (embeddings_bench.jl)
# - Embedding optimization (optimize_bench.jl)
# - Integration (fit) benchmarks
#
# See PLAN.md for details on the benchmarking strategy.

using BenchmarkTools
using UMAP
import NearestNeighborDescent as NND
using Random

include("utils.jl")
include("simplicial_sets_bench.jl")
include("embeddings_bench.jl")
include("optimize_bench.jl")

suite = BenchmarkGroup()

# Integration benchmarks for UMAP.fit
suite["fit"] = BenchmarkGroup(["integration"])

const N_POINTS = [1000, 10_000]
const IN_DIMS = [10, 50]
const OUT_DIMS = [2, 10]

for (n_points, in_dims, out_dims) in Iterators.product(N_POINTS, IN_DIMS, OUT_DIMS)
    data = matrix_data(n_points, in_dims)
    suite["fit"]["matrix"]["$(n_points)x$(in_dims)x$(out_dims)"] = @benchmarkable(
        UMAP.fit($data, $out_dims; n_neighbors=15, n_epochs=100),
        setup=(GC.gc()),
        evals=1,
        samples=10
    )

    vec_data = vecvec_data(n_points, in_dims)
    suite["fit"]["vectors"]["$(n_points)x$(in_dims)x$(out_dims)"] = @benchmarkable(
        UMAP.fit($vec_data, $out_dims; n_neighbors=15, n_epochs=100),
        setup=(GC.gc()),
        evals=1,
        samples=10
    )
end

# Add component benchmark suites
suite["simplicial"] = SimplicialBench.simplicial_suite
suite["embeddings"] = EmbeddingsBench.embeddings_suite
suite["optimize"] = OptimizeBench.optimize_suite

# Only tune and run when executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    tune!(suite)
    results = run(suite, verbose=true, seconds=1)
    print(results)
end
