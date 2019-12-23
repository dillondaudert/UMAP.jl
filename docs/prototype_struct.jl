# prototype of rework for umap infrastructure using structs instead of
# named tuples to parameterize the various subroutines of the UMAP
# algorithm

# some assumptions:
# 1. UMAP doesn't need to have parameterizable structs that can be taken by
#    users that they then customize usage themselves (whoo that grammar...).
#    What I mean is all different ways of paramaterizing the algorithm will be supported
#    explicitly in the package, so the design doesn't need to try to enable 3rd party
#    customization; all supported functionality will exist in this repo
#  => this imo means we don't need the flexibility of dispatching everything on named tuples
#     representing arbitrary parametizations... I think the work required would lead to very
#     little benefit.

#############################
# PARAMETER STRUCTS

struct NNeighborParameters
    method
    k
    metric
    kwargs
end

struct SimplicialSetParameters
    local_connectivity
    merge_op_ratio
end

struct EmbeddingParameters
    metric
    n_components
    init
    min_dist
    spread
    a
    b
end


struct OptimizationParameters
    n_epochs
    learning_rate
    repulsion_strength
    neg_sample_rate
end

#########################
# knn subroutines

function knn_search(data, )
