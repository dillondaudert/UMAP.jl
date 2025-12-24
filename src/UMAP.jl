module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using NearestNeighborDescent
using Setfield
using SparseArrays

include("utils.jl")
include("membership_fn.jl")
include("neighbors.jl")
include("simplicial_sets.jl")
include("embeddings.jl")
include("optimize.jl")
include("config.jl")
include("fit.jl")
include("transform.jl")

public fit, transform

end # module
