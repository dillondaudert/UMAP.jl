
function transform(result::UMAPResult,
                   queries,
                   knn_params,
                   src_params,
                   gbl_params,
                   tgt_params,
                   opt_params)
    
    # 1. find (approx) nearest neighbors to `queries` in `result.data`
    knns_dists = knn_search(result.data, queries, knn_params, result.knns_dists)
    # 2. compute fuzzy simpl sets and construct the global umap graph
    fs_sets = fuzzy_simplicial_set(result.data, knns_dists, knn_params, src_params)
    query_graph = coalesce_views(fs_sets, gbl_params)
    # 3. initialize target embedding for queries
    query_embedding = initialize_embedding(query_graph, tgt_params)
    # 4. optimize query embedding
    optimize_embedding!(query_embedding, result.embedding, query_graph, tgt_params, opt_params)

    return UMAPTransformResult(queries, query_embedding, knns_dists, fs_sets, query_graph)
end