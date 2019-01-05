module UMAP

using Distances: evaluate, Euclidean, SqEuclidean, SemiMetric
using NearestNeighborDescent: DescentGraph
using LsqFit: curve_fit
using Random: shuffle!
using SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, dropzeros, nzrange, rowvals, nonzeros
using LinearAlgebra: Symmetric, Diagonal, issymmetric, I
using Arpack: eigs

include("utils.jl")
include("umap_.jl")

export UMAP_

end # module
