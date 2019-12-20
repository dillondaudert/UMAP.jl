## Function and the arguments they use

`fuzzy_simplicial_set(X, n_neighbors, metric, local_connectivity, set_operation_ratio)`
- outputs a weighted `graph` (usually a sparse matrix)

`initialize_embedding(graph, n_components, init)`
- outputs an `embedding` that is to be optimized

`fit_ab(min_dist, spread)`
- `a`, `b` are outputs of `fit_ab` OR passed as arguments

`optimize_embedding!(embedding, graph, n_epochs, learning_rate, repulsion_strength, neg_sample_rate, a, b)`

### Subfunctions and their args

`fuzzy_simplicial_set(X, n_neighbors, metric, local_connectivity, set_operation_ratio)`
- `knn_search(X, n_neighbors, metric) -> knns, dists`
- `smooth_knn_dists(dists, n_neighbors, local_connectivity) -> sigmas, rhos`
- `compute_membership_strengths(knns, dists, sigmas, rhos) -> fs_set`
- `combine_fuzzy_sets(fs_set, set_operation_ratio) -> combined_fs_set`


## Arguments and where they are used (current names)

Format: `arg`: `<used in>`; `<used in>()`

`X`: fuzzy_simplicial_set,

`n_components`: initialize_embedding,

`n_neighbors`: fuzzy_simplicial_set,

`metric`: fuzzy_simplicial_set,

`n_epochs`: optimize_embedding

`learning_rate`: optimize_embedding

`init`: initialize_embedding,

`min_dist`: fit_ab

`spread`: fit_ab

`set_operation_ratio`: fuzzy_simplicial_set,

`local_connectivity`: fuzzy_simplicial_set,

`repulsion_strength`: optimize_embedding

`neg_sample_rate`: optimize_embedding

`a`, `b`: optimize_embedding
