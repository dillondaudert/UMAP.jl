# Benchmarks for fuzzy simplicial set construction

module SimplicialBench

using BenchmarkTools
using UMAP
using Distances
using SparseArrays
import NearestNeighborDescent as NND

include("utils.jl")

const N_POINTS = [1_000, 10_000]
const IN_DIMS = [10, 50]
const KNN = [5, 15, 50]

simplicial_suite = BenchmarkGroup(["simplicial"])
simplicial_suite["fuzzy_simplicial_set"] = BenchmarkGroup(["runtime"])
simplicial_suite["smooth_knn_dists"] = BenchmarkGroup(["runtime"])
simplicial_suite["compute_membership_strengths"] = BenchmarkGroup(["runtime"])
simplicial_suite["merge_local_simplicial_sets"] = BenchmarkGroup(["runtime"])

# Top-level fuzzy_simplicial_set benchmarks
for (n_points, in_dims, knn) in Iterators.product(N_POINTS, IN_DIMS, KNN)
    knns, dists = knns_dists(n_points, in_dims, knn)
    knn_params = UMAP.DescentNeighbors(knn, Euclidean())
    src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)

    simplicial_suite["fuzzy_simplicial_set"]["$(n_points)x$(in_dims)_k$(knn)"] = @benchmarkable(
        UMAP.fuzzy_simplicial_set(($knns, $dists), $n_points, $knn_params, $src_params, true),
        setup=(GC.gc()),
        evals=2,
        samples=20,
    )
end

# smooth_knn_dists benchmarks
for (n_points, in_dims, knn) in Iterators.product(N_POINTS, IN_DIMS, KNN)
    _, dists = knns_dists(n_points, in_dims, knn)
    src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)

    simplicial_suite["smooth_knn_dists"]["$(n_points)x$(in_dims)_k$(knn)"] = @benchmarkable(
        UMAP.smooth_knn_dists($dists, $knn, $src_params),
        setup=(GC.gc()),
        evals=2,
        samples=20,
    )
end

# compute_membership_strengths benchmarks
for (n_points, in_dims, knn) in Iterators.product(N_POINTS, IN_DIMS, KNN)
    knns, dists = knns_dists(n_points, in_dims, knn)
    src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)
    # Pre-compute σs and ρs
    ρs, σs = UMAP.smooth_knn_dists(dists, knn, src_params)

    simplicial_suite["compute_membership_strengths"]["$(n_points)x$(in_dims)_k$(knn)"] = @benchmarkable(
        UMAP.compute_membership_strengths($knns, $dists, $ρs, $σs),
        setup=(GC.gc()),
        evals=2,
        samples=20,
    )
end

# merge_local_simplicial_sets benchmarks
for (n_points, in_dims, knn) in Iterators.product(N_POINTS, IN_DIMS, KNN)
    knns, dists = knns_dists(n_points, in_dims, knn)
    src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)
    # Pre-compute the local fuzzy simplicial sets
    ρs, σs = UMAP.smooth_knn_dists(dists, knn, src_params)
    rows, cols, vals = UMAP.compute_membership_strengths(knns, dists, ρs, σs)
    local_fs_sets = sparse(rows, cols, vals, n_points, n_points)

    # Test with different set_op_ratios
    for set_op_ratio in [0.0, 0.5, 1.0]
        simplicial_suite["merge_local_simplicial_sets"]["$(n_points)x$(in_dims)_k$(knn)_ratio$(set_op_ratio)"] = @benchmarkable(
            UMAP.merge_local_simplicial_sets($local_fs_sets, $set_op_ratio),
            setup=(GC.gc()),
            evals=2,
            samples=20,
        )
    end
end

end # module
