
module SimplicialBench
using BenchmarkTools
using UMAP
using Distances
import NearestNeighborDescent as NND
include("utils.jl")

const N_POINTS = [10_000,]
const IN_DIMS = [10, 50]
const KNN = [5, 10, 50]

simplicial_suite = BenchmarkGroup(["simplicial"])

for (n_points, in_dims, knn) in Iterators.product(N_POINTS, IN_DIMS, KNN)

    knns, dists = knns_dists(n_points, in_dims, knn)
    knn_params = UMAP.DescentNeighbors(knn, Euclidean())
    src_params = UMAP.SourceViewParams(1, 1, 1)

    simplicial_suite["$(n_points)x$(in_dims)x$(knn)"] = @benchmarkable(
        UMAP.fuzzy_simplicial_set(($knns, $dists), $n_points, $knn_params, $src_params, true),
        setup=(GC.gc()),
        evals=2,
        samples=20,
    )

end



end