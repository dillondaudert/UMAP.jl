"""
    transform(result::UMAPResult, queries) -> UMAPTransformResult

Use the given UMAP result to embed new points into an existing embedding. `queries` is a matrix or vector of some number of points
in the same space as `result.data`. The returned embedding is the embedding of these points in n-dimensional space, where
n is the dimensionality of `result.embedding`. This embedding is created by finding neighbors of `queries` in `result.embedding`
and optimizing cross entropy according to membership strengths according to these neighbors.

The transform is parameterized by the config found in `result`. For that reason, the type of 
`result` must match exactly `result.data` - including as a named tuple if necessary.
"""
function transform(
        result::U,
        queries::DS
    ) where {DS, U <: UMAPResult{DS}}

    knn_params = result.config.knn_params
    src_params = result.config.src_params
    gbl_params = result.config.gbl_params
    tgt_params = result.config.tgt_params
    opt_params = result.config.opt_params

    # default opt params modified slightly for transform.
    opt_params = Accessors.@set opt_params.n_epochs = 30
    opt_params = Accessors.@set opt_params.lr /= 4

    # NOTE: pyumap sets local_connectivity - 1

    return transform(result, queries, knn_params, src_params, gbl_params, tgt_params, opt_params)

end

"""
    transform(result::UMAPResult, queries, knn_params, src_params, gbl_params, tgt_params, opt_params)

Transform the UMAP result for new queries.
This method allows overriding the transform-time parameters by passing in configuration structs
directly.
"""
function transform(
        result::U,
        queries::DS,
        knn_params,
        src_params,
        gbl_params,
        tgt_params,
        opt_params
    ) where {DS, U <: UMAPResult{DS}}

    # 1. find (approx) nearest neighbors to `queries` in `result.data`
    knns_dists = knn_search(result.data, queries, knn_params, result.knns_dists)
    # 2. compute fuzzy simpl sets and construct the global umap graph
    fs_sets = fuzzy_simplicial_set(result.data, knns_dists, knn_params, src_params)
    query_graph = coalesce_views(fs_sets, gbl_params)
    # 3. initialize target embedding for queries
    query_embedding = initialize_embedding(result.embedding, query_graph, tgt_params)
    # 4. optimize query embedding
    optimize_embedding!(query_embedding, result.embedding, query_graph, tgt_params, opt_params)

    return UMAPTransformResult(queries, query_embedding, knns_dists, fs_sets, query_graph)
end
