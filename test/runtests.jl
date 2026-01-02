using Test
using Random
using SparseArrays
using LinearAlgebra
import UMAP
import Distances
import NearestNeighborDescent
import DifferentiationInterface as DI
import Zygote

include("utils_tests.jl")
include("neighbors_tests.jl")
include("simplicial_sets_tests.jl")
include("membership_fn_tests.jl")
include("embeddings_tests.jl")
include("optimize_tests.jl")
include("config_tests.jl")
#include("fit_tests.jl")
