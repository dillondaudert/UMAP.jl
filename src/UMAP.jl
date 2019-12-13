module UMAP

using Arpack
using Distances
using LinearAlgebra
using LsqFit: curve_fit
using NearestNeighborDescent
using SparseArrays
using StatsBase

include("utils.jl")
include("layouts.jl")
include("umap_.jl")

export umap, UMAP_

end # module
