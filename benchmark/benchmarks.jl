# Benchmark suite for UMAP.jl
#
# This suite covers the major UMAP algorithm stages:
# - Fuzzy simplicial set construction (simplicial_sets_bench.jl)
# - Embedding initialization (embeddings_bench.jl)
# - Embedding optimization (optimize_bench.jl)
# - Integration benchmarks (fit, transform)
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

SUITE = BenchmarkGroup()

# Integration benchmarks for UMAP.fit
SUITE["fit"] = BenchmarkGroup(["integration"])

const N_POINTS = [1000, 10_000]
const IN_DIMS = [10, 50]
const OUT_DIMS = [2, 10]

for (n_points, in_dims, out_dims) in Iterators.product(N_POINTS, IN_DIMS, OUT_DIMS)
    data = matrix_data(n_points, in_dims)
    SUITE["fit"]["matrix"]["$(n_points)x$(in_dims)x$(out_dims)"] = @benchmarkable(
        UMAP.fit($data, $out_dims; n_neighbors=15, n_epochs=100),
        setup=(GC.gc()),
        evals=1,
        samples=10
    )

    vec_data = vecvec_data(n_points, in_dims)
    SUITE["fit"]["vectors"]["$(n_points)x$(in_dims)x$(out_dims)"] = @benchmarkable(
        UMAP.fit($vec_data, $out_dims; n_neighbors=15, n_epochs=100),
        setup=(GC.gc()),
        evals=1,
        samples=10
    )
end

# Integration benchmarks for UMAP.transform
SUITE["transform"] = BenchmarkGroup(["integration"])

# For transform, we fit on a base dataset first, then transform new queries
const N_QUERY_POINTS = [100, 1000]

for (n_points, in_dims, out_dims) in Iterators.product(N_POINTS, IN_DIMS, OUT_DIMS)
    # Pre-fit the UMAP model on base data
    base_data = matrix_data(n_points, in_dims; seed=111222333)
    result = UMAP.fit(base_data, out_dims; n_neighbors=15, n_epochs=100)

    for n_queries in N_QUERY_POINTS
        query_data = matrix_data(n_queries, in_dims; seed=444555666)
        SUITE["transform"]["matrix"]["$(n_points)_base_$(n_queries)_query_$(in_dims)x$(out_dims)"] = @benchmarkable(
            UMAP.transform($result, $query_data),
            setup=(GC.gc()),
            evals=1,
            samples=10
        )

        query_vec_data = vecvec_data(n_queries, in_dims; seed=444555666)
        vec_result = UMAP.fit(vecvec_data(n_points, in_dims; seed=111222333), out_dims; n_neighbors=15, n_epochs=100)
        SUITE["transform"]["vectors"]["$(n_points)_base_$(n_queries)_query_$(in_dims)x$(out_dims)"] = @benchmarkable(
            UMAP.transform($vec_result, $query_vec_data),
            setup=(GC.gc()),
            evals=1,
            samples=10
        )
    end
end

# Add component benchmark suites
SUITE["simplicial"] = SimplicialBench.simplicial_suite
SUITE["embeddings"] = EmbeddingsBench.embeddings_suite
SUITE["optimize"] = OptimizeBench.optimize_suite

# Only tune and run when executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    tune!(SUITE)
    results = run(SUITE, verbose=true, seconds=1)
    print(results)
end
