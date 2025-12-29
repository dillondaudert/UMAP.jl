


"""
    transform(result::UMAPResult, queries; <kwargs>) -> UMAPTransformResult

Use the given UMAP result to embed new points into an existing embedding. `queries` is a matrix or vector of some number of points
in the same space as `result.data`. The returned embedding is the embedding of these points in n-dimensional space, where
n is the dimensionality of `result.embedding`. This embedding is created by finding neighbors of `queries` in `result.embedding`
and optimizing cross entropy according to membership strengths according to these neighbors.

# Keyword Arguments
- `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
- `metric::{SemiMetric, Symbol} = Euclidean()`: the metric to calculate distance in the input space. It is also possible to pass `metric = :precomputed` to treat `X` like a precomputed distance matrix.
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `init::Symbol = :spectral`: how to initialize the output embedding; valid options are `:spectral` and `:random`
- `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
- `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `a = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `b = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
"""
# TODO

"""
    transform(result::UMAPResult, queries, knn_params, src_params, gbl_params, tgt_params, opt_params)

Transform the UMAP result for new queries.
This function takes the result of a UMAP fit and applies it to new queries, returning an `UMAPTransformResult`.
"""
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
    query_embedding = initialize_embedding(result.embedding, query_graph, tgt_params)
    # 4. optimize query embedding
    optimize_embedding!(query_embedding, result.embedding, query_graph, tgt_params, opt_params)

    return UMAPTransformResult(queries, query_embedding, knns_dists, fs_sets, query_graph)
end