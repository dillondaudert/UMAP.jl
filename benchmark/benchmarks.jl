# benchmark suite for UMAP.jl

using BenchmarkTools
using UMAP
import NearestNeighborDescent as NND
using Random

include("utils.jl")
include("simplicial_sets_bench.jl")

suite = BenchmarkGroup()

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

suite["simplicial"] = SimplicialBench.simplicial_suite

tune!(suite)

results = run(suite, verbose=true, seconds=1)
print(results)