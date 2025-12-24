# parameter structs for configuring UMAP

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
        n_epochs > 0 || throw(ArgumentError("n_epochs must be greater than 0, got $n_epochs"))
        lr ≥ 0 || throw(ArgumentError("learning_rate must be non-negative, got $lr"))
        neg_sample_rate ≥ 0 || throw(ArgumentError("neg_sample_rate must be non-negative, got $neg_sample_rate"))
        return new(n_epochs, lr, repulsion_strength, neg_sample_rate)
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
