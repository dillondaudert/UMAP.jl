# parameter structs for configuring UMAP


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
