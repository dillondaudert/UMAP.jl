module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using NearestNeighborDescent
using SparseArrays

include("utils.jl")
include("embeddings.jl")
include("umap_.jl")

export umap, UMAP_

end # module
