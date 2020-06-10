# parameter structs for configuring UMAP

# NEIGHBOR PARAMS
"""
Structs for parameterizing the knn search step of UMAP.
"""
abstract type NeighborParams end

# for finding approximate nearest neighbors
struct DescentNeighbors{M, K} <: NeighborParams
    n_neighbors::Int
    metric::M
    kwargs::K
end
DescentNeighbors(n_neighbors, metric) = DescentNeighbors(n_neighbors, metric, NamedTuple())

# for precomputed distance matrix
struct PrecomputedNeighbors{M} <: NeighborParams
    n_neighbors::Int
    dists::M
end

# SOURCE PARAMS
"""
Struct for parameterizing the representation of the data in the source (original)
manifold; i.e. constructing fuzzy simplicial sets of each view of the dataset.
"""
struct SourceViewParams
    set_operation_ratio::Float64
    local_connectivity::Float64
    bandwidth::Float64
end

"""
Parameters for merging the fuzzy simplicial sets for each dataset view into one
fuzzy simplicial set.
"""
struct SourceGlobalParams
    mix_ratio::Float64
end

"""
Configuration struct for the UMAP algorithm.
"""
struct UMAPConfig{K, S, G, T, O}
    knn_params::K
    src_params::S
    gbl_params::G
    tgt_params::T
    opt_params::O
end

"""
View-specific results.
"""
struct UMAPViewResult{K, D, R, S}
    knns::K
    dists::D
    rhos::R
    sigmas::S
end


"""
Return result of the UMAP algorithm.
"""
struct UMAPResult{DS, DT, C, V}
    data::DS
    embedding::DT
    config::C
    views::V
end
