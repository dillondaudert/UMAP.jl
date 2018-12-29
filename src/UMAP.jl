module UMAP

using NearestNeighborDescent: DescentGraph
using SparseArrays
using LinearAlgebra
using Arpack

include("umap_.jl")

export UMAP_

end # module
