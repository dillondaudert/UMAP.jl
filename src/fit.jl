

"""
    function fit(data[, n_components=2]; <kwargs>) -> UMAPResult

Embed `data` into a `n_components`-dimensional space. Returns a `UMAPResult`.

# Keyword Arguments
- `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
- `metric::{SemiMetric, Symbol} = Euclidean()`: the metric to calculate distance in the input space. It is also possible to pass `metric = :precomputed` to treat `data` like a precomputed distance matrix.
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `init::AbstractInitialization = UMAPA.SpectralInitialization()`: how to initialize the output embedding; valid options are `UMAP.SpectralInitialization()` and `UMAP.UniformInitialization()`
- `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
- `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
"""
function fit(data,
             n_components = 2;
             n_neighbors = 15,
             metric = Distances.Euclidean(),
             n_epochs = 300,
             learning_rate = 1,
             init::AbstractInitialization = SpectralInitialization(),
             min_dist = 0.1,
             spread = 1.,
             set_operation_ratio = 1.,
             local_connectivity = 1.,
             bandwidth = 1.,
             repulsion_strength = 1.,
             neg_sample_rate = 5
)
    # create config from kw args using helper
    config = create_config(
        # view-specific parameters (knn_search, source view params)
        (
            data_or_dists=data,
            n_neighbors=n_neighbors,
            metric=metric,
            knn_kwargs=NamedTuple(),
            set_operation_ratio=set_operation_ratio,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth
        );
        # global, target, optimization parameters
        global_mix_ratio=0.5, # not used for single view in this case
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        target_metric=Distances.SqEuclidean(),
        init=init,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        repulsion_strength=repulsion_strength,
        neg_sample_rate=neg_sample_rate
    )

    return fit(config...)

end

# the generic fit algorithm - works for single and  named tuples of configs
function fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)

    # 1. find (approx) nearest neighbors
    knns_dists = knn_search(data, knn_params)
    # 2. construct the umap graph (global fuzzy simplicial set)
    fs_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_params)
    umap_graph = coalesce_views(fs_sets, gbl_params)
    # 3. initialize the target embedding
    embedding = initialize_embedding(umap_graph, tgt_params)
    # 4. optimize the embedding
    optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)

    config = UMAPConfig(knn_params, src_params, gbl_params, tgt_params, opt_params)
    return UMAPResult(data, embedding, config, knns_dists, fs_sets, umap_graph)
end


