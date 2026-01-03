module UMAP

# for stdlibs, we use:
using LinearAlgebra
using SparseArrays
# for other packages, we use import to keep namespaces clear
import Arpack
import Distances
import LsqFit
import NearestNeighborDescent as NND
import Accessors

include("utils.jl")
include("membership_fn.jl")
include("neighbors.jl")
include("simplicial_sets.jl")
include("embeddings.jl")
include("optimize.jl")
include("config.jl")
include("fit.jl")
include("transform.jl")

public fit, transform, UMAPConfig, UMAPResult, UMAPTransformResult

end # module
