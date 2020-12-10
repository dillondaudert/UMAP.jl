
function fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)

    # 1. find (approx) nearest neighbors
    knns_dists = knn_search(data, knn_params)
    # 2. construct the umap graph (global fuzzy simplicial set)
    fs_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_params)
    umap_graph = coalesce_views(fs_sets, gbl_params)
    # 3. initialize the target embedding
    embedding = initialize_embedding(umap_graph, tgt_params)
    # 4. optimize the embedding
    optimize_embedding!(umap_graph, opt_params)

    config = UMAPConfig(knn_params, src_params, gbl_params, tgt_params, opt_params)
    return UMAPResult(data, embedding, config, knns_dists, fs_sets, umap_graph)
end


