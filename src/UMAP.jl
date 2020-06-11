module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using NearestNeighborDescent
using SparseArrays

include("utils.jl")
include("umap_.jl")
include("config.jl")
include("neighbors.jl")
include("simplicial_sets.jl")
include("embeddings.jl")
include("optimize.jl")

export umap, UMAP_, transform

end # module
