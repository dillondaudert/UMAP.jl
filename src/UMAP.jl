module UMAP

using Distances: evaluate, Euclidean, SquaredEuclidean, SemiMetric
using NearestNeighborDescent: DescentGraph
using LsqFit: curve_fit
using SparseArrays: SparseMatrixCSC, sparse, dropzeros, nzrange, rowvals, nonzeros
using LinearAlgebra: Symmetric, Diagonal, issymmetric, I
using Arpack: eigs

include("umap_.jl")

export UMAP_

end # module
