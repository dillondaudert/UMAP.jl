using Test
using Distances: Euclidean, CosineDist
using Random
using SparseArrays
using LinearAlgebra
using UMAP
using UMAP: initialize_embedding, fuzzy_simplicial_set, compute_membership_strengths, smooth_knn_dists, smooth_knn_dist, spectral_layout, knn_search, combine_fuzzy_sets, fit_ab, SMOOTH_K_TOLERANCE
using UMAP: DescentNeighbors, PrecomputedNeighbors
using UMAP: SourceViewParams, SourceGlobalParams
using UMAP: coalesce_views
using UMAP: fit_ab

include("neighbors_tests.jl")
include("simplicial_sets_tests.jl")
include("membership_fn_tests.jl")
