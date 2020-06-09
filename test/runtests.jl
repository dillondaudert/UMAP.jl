using Test
using Distances: Euclidean, CosineDist
using Random
using SparseArrays
using LinearAlgebra
using UMAP
using UMAP: initialize_embedding, fuzzy_simplicial_set, compute_membership_strengths, smooth_knn_dists, smooth_knn_dist, spectral_layout, optimize_embedding, knn_search, combine_fuzzy_sets, fit_ab, SMOOTH_K_TOLERANCE
using UMAP: DescentNeighbors, PrecomputedNeighbors
using UMAP: SourceViewParams, SourceGlobalParams

include("neighbors_tests.jl")
include("simplicial_sets_tests.jl")
