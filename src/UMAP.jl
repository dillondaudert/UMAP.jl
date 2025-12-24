module UMAP

# for stdlibs, we use:
using LinearAlgebra
using SparseArrays
# for other packages, we use import to keep namespaces clear
import Arpack
using Distances
using LsqFit: curve_fit
using NearestNeighborDescent
using Setfield

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
