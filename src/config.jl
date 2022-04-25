# parameter structs for configuring UMAP

# CONSTANTS
const SMOOTH_K_TOLERANCE = 1e-5

# NEIGHBOR PARAMS
"""
Structs for parameterizing the knn search step of UMAP.
"""
abstract type NeighborParams end

"""
    DescentNeighbors(n_neighbors, metric, kwargs)

Parameters for finding approximate nearest neighbors using NearestNeighborDescent.
"""
struct DescentNeighbors{M, K} <: NeighborParams
    "The number of neighbors to consider as connected. The more neighbors, the more global structure is captured in the data."
    n_neighbors::Int
    "A distance function for comparing two points"
    metric::M
    "Keyword arguments to pass to NearestNeighborDescent.nndescent()"
    kwargs::K
end
DescentNeighbors(n_neighbors, metric) = DescentNeighbors(n_neighbors, metric, NamedTuple())

"""
    PrecomputedNeighbors(n_neighbors, dists)

Parameters for finding nearest neighbors from precomputed distances.
"""
struct PrecomputedNeighbors{M} <: NeighborParams
    n_neighbors::Int
    dists::M
end

# SOURCE PARAMS
"""
    SourceViewParams(set_operation_ratio, local_connectivity, bandwidth)

Struct for parameterizing the representation of the data in the source (original)
manifold; i.e. constructing fuzzy simplicial sets of each view of the dataset.
"""
struct SourceViewParams{T<:Real}
    """
    The ratio of set union to set intersection used to combine local fuzzy simplicial sets, 
    from 0 (100% intersection) to 1 (100% union)
    """
    set_operation_ratio::T
    """
    The number of nearest neighbors that should be assumed to be locally connected. 
    The higher this value, the more connected the manifold becomes. 
    This should not be set higher than the intrinsic dimension of the manifold.
    """
    local_connectivity::T
    "bandwidth"
    bandwidth::T
    function SourceViewParams{T}(set_op_ratio, local_conn, bandwidth) where {T <: Real}
        0 ≤ set_op_ratio ≤ 1 || error("set_op_ratio must be between 0 and 1")
        local_conn > 0 || error("local_connectivity must be greater than 0")
        bandwidth > 0 || error("bandwidth must be greater than 0")
        new(set_op_ratio, local_conn, bandwidth)
    end
end
function SourceViewParams(set_op_ratio::T, local_conn::T, bandwidth::T) where {T <: Real}
    SourceViewParams{T}(set_op_ratio, local_conn, bandwidth)
end
function SourceViewParams(set_op_ratio::Real, local_conn::Real, bandwidth::Real)
    SourceViewParams(promote(set_op_ratio, local_conn, bandwidth)...)
end

"""
    SourceGlobalParams{T}(mix_ratio)

Parameters for merging the fuzzy simplicial sets for each dataset view into one
fuzzy simplicial set, otherwise known as the UMAP graph.
"""
struct SourceGlobalParams{T <: Real}
    mix_ratio::T
    function SourceGlobalParams{T}(mix_ratio) where {T <: Real}
        0 ≤ mix_ratio ≤ 1 || error("mix_ratio must be between 0 and 1")
        new(mix_ratio)
    end
end
SourceGlobalParams(mix_ratio::T) where {T <: Real} = SourceGlobalParams{T}(mix_ratio)

"""
    TargetParams{M, D, I, P}(manifold::M, metric::D, init::I, memb_params::P)

Parameters for controlling the target embedding, e.g. the manifold, distance metric, initialization 
method.
"""
struct TargetParams{M, D, I, P}
    "The target manifold in which to embed the data"
    manifold::M
    "The metric used to compute distances on the target manifold"
    metric::D
    "The method of initialization for points on the target manifold"
    init::I
    "Parameters for the membership function of the target embedding (see MembershipFnParams)"
    memb_params::P
end

"""
    MembershipFnParams{T}(min_dist, spread, a, b)
"""
mutable struct MembershipFnParams{T <: Real}
    "The minimum spacing of points in the target embedding"
    min_dist::T
    "The effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`."
    spread::T
    a::T
    b::T
    function MembershipFnParams{T}(min_dist, spread, a, b) where {T <: Real}
        min_dist > 0 || error("min_dist must be greater than 0")
        spread > 0 || error("spread must be greater than 0")
        new(min_dist, spread, a, b)
    end
end
function MembershipFnParams(min_dist::T, spread::T, a::T, b::T) where {T <: Real}
    MembershipFnParams{T}(min_dist, spread, a, b)
end
# autopromote
function MembershipFnParams(min_dist::Real, spread::Real, a::Real, b::Real)
    MembershipFnParams(promote(min_dist, spread, a, b)...)
end
# calculate a, b with binary search
function MembershipFnParams(min_dist::Real, spread::Real, ::Nothing, ::Nothing)
    a, b = fit_ab(min_dist, spread)
    MembershipFnParams(min_dist, spread, a, b)
end
function MembershipFnParams(min_dist::Real, spread::Real)
    MembershipFnParams(min_dist, spread, nothing, nothing)
end

"""
    OptimizationParams(n_epochs, learning_rate, repulsion_strength, neg_sample_rate)

Parameters for controlling the optimization process.
"""
struct OptimizationParams
    "The number of epochs to perform optimization"
    n_epochs::Int
    "The initial learning rate for optimization (decreases each epoch)"
    lr::Float64
    "The weighting of negative samples during optimization"
    repulsion_strength::Float64
    """
    The number of negative samples to select for each positive sample.
    Higher values increase computational cost but result in slightly better accuracy.
    """
    neg_sample_rate::Int
    function OptimizationParams(n_epochs, lr, repulsion_strength, neg_sample_rate)
        n_epochs > 0 || error("n_epochs must be greater than 0, got $n_epochs")
        lr ≥ 0 || error("learning_rate must be non-negative, got $lr")
        neg_sample_rate ≥ 0 || error("neg_sample_rate must be non-negative, got $neg_sample_rate")
        new(n_epochs, lr, repulsion_strength, neg_sample_rate)
    end
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
Return result of the UMAP algorithm.
"""
struct UMAPResult{DS, DT, C, K, F, G}
    data::DS
    embedding::DT
    config::C
    knns_dists::K
    fs_sets::F
    graph::G
end

struct UMAPTransformResult{DS, DT, K, F, G}
    data::DS
    embedding::DT
    knns_dists::K
    fs_sets::F
    graph::G
end