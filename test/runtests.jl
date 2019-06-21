using Test
using Distances: Euclidean, CosineDist
using SparseArrays
using LinearAlgebra
using UMAP
using UMAP: fuzzy_simplicial_set, compute_membership_strengths, smooth_knn_dists, smooth_knn_dist, spectral_layout, optimize_embedding, knn_search, combine_fuzzy_sets, fit_ab, SMOOTH_K_TOLERANCE


include("utils_tests.jl")
include("umap_tests.jl")
