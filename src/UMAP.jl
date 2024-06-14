module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using NearestNeighborDescent
using SparseArrays

include("umap_.jl")
include("utils.jl")
include("embeddings.jl")

export Embedding, Categorical, DataView

end # module
