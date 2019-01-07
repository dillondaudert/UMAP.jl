using Test
using Distances: Euclidean
using SparseArrays
using LinearAlgebra
using UMAP
using UMAP: fuzzy_simplicial_set, compute_membership_strengths, smooth_knn_dists, smooth_knn_dist, spectral_layout, optimize_embedding, pairwise_knn

include("umap_tests.jl")