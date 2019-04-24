module UMAP

using Distances: evaluate, pairwise!, Euclidean, SqEuclidean, SemiMetric
using NearestNeighborDescent: DescentGraph
using LsqFit: curve_fit
using SparseArrays: SparseMatrixCSC, sparse, dropzeros, nzrange, rowvals, nonzeros
using LinearAlgebra: Symmetric, Diagonal, issymmetric, I
using Arpack: eigs

include("utils.jl")
include("umap_.jl")

export umap, UMAP_, Precomputed

end # module
