# struct and utilities for the UMAP algorithm configuration.


ViewConfigNT = NamedTuple{(
    :data_or_dists, 
    :n_neighbors,
    :metric,
    :knn_kwargs,
    :set_operation_ratio,
    :local_connectivity,
    :bandwidth
)}

"""
Utility for creating the configuration structs for UMAP.
"""
function create_config end

function create_config(
    view_params::Vector{NT};
    # global simplicial set params
    global_mix_ratio,
    # target params
    min_dist,
    spread,
    n_components,
    target_metric,
    init,
    # optimization params 
    n_epochs,
    learning_rate,
    repulsion_strength,
    neg_sample_rate
) where {NT <: ViewConfigNT}

    view_configs = map(params -> create_view_config(;pairs(params)...), view_params)
    if length(view_configs) == 1
        data_param = view_params[1].data_or_dists
        knn_params, src_params = view_configs[1]
    else
        # create named tuple of view params by view
        view_keys = ["view_$i" for i in eachindex(view_params)]
        data_param = ([Symbol(view_keys[i])=>view_params[i].data_or_dists for i in eachindex(view_params)]...,)
        knn_params = ([Symbol(view_keys[i])=>view_configs[i][1] for i in eachindex(view_configs)]...,)
        src_params = ([Symbol(view_keys[i])=>view_configs[i][2] for i in eachindex(view_configs)]...,)
    end

    gbl_params = SourceGlobalParams(global_mix_ratio)

    memb_params = MembershipFnParams(min_dist, spread)
    tgt_params = TargetParams(_EuclideanManifold(n_components), target_metric, init, memb_params)

    opt_params = OptimizationParams(n_epochs, learning_rate, repulsion_strength, neg_sample_rate)

    return (data_param, knn_params, src_params, gbl_params, tgt_params, opt_params)
end

"""
Utility for creating the view-specific structs for UMAP, i.e.
NeighborParams and SourceViewParams.
"""
function create_view_config end

"""
Create new NeighborParams, SourceViewParams for a view of the data 
given the arguments.
"""
function create_view_config(;
    data_or_dists,
    n_neighbors,
    metric,
    set_operation_ratio,
    local_connectivity,
    bandwidth,
    knn_kwargs = NamedTuple()
)
    # KNN PARAMS 
    if metric == :precomputed
        knn_params = PrecomputedNeighbors(n_neighbors, data_or_dists)
    else
        knn_params = DescentNeighbors(n_neighbors, metric, knn_kwargs)
    end

    src_params = SourceViewParams(set_operation_ratio, local_connectivity, bandwidth)

    return (knn_params, src_params)
end

"""
Create NeighborParams, SourceViewParams for a view of the data based on 
the params previously used to fit this same view. Parameters can be 
overridden by keyword arguments.
"""
function create_view_config(
    fit_knn_params::NeighborParams,
    fit_src_params::SourceViewParams;
    data_or_dists,
    n_neighbors=nothing,
    metric=nothing,
    set_operation_ratio=nothing,
    local_connectivity=nothing,
    bandwidth=nothing,
    knn_kwargs=nothing
)
    # KNN PARAMS 
    if fit_knn_params isa PrecomputedNeighbors
        knn_params = PrecomputedNeighbors(
            something(n_neighbors, fit_knn_params.n_neighbors), 
            data_or_dists
        )
    else
        knn_params = DescentNeighbors(
            something(n_neighbors, fit_knn_params.n_neighbors),
            something(metric, fit_knn_params.metric),
            something(knn_kwargs, fit_knn_params.kwargs)
        )
    end

    # SRC VIEW PARAMS 
    src_params = SourceViewParams(
        something(set_operation_ratio, fit_src_params.set_operation_ratio),
        something(local_connectivity, fit_src_params.local_connectivity),
        something(bandwidth, fit_src_params.bandwidth)
    )
    
    return (knn_params, src_params)

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
