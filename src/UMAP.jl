module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using NearestNeighborDescent
using Setfield
using SparseArrays

include("utils.jl")
include("umap_.jl")
include("config.jl")
include("neighbors.jl")
include("simplicial_sets.jl")
include("embeddings.jl")
include("membership_fn.jl")
include("optimize_old.jl")
include("optimize_new.jl")
include("fit.jl")

export umap, UMAP_, transform

end # module
