module UMAP

using NearestNeighborDescent: DescentGraph
using LsqFit: curve_fit
using SparseArrays
using LinearAlgebra
using Arpack: eigs

include("umap_.jl")

export UMAP_

end # module
