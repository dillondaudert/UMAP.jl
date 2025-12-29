using Test
using Random
using SparseArrays
using LinearAlgebra
import UMAP
import Distances
import NearestNeighborDescent

include("config_tests.jl")
include("utils_tests.jl")
include("neighbors_tests.jl")
include("simplicial_sets_tests.jl")
include("membership_fn_tests.jl")
include("embeddings_tests.jl")
#include("optimize_tests.jl")
#include("umap_tests.jl")
