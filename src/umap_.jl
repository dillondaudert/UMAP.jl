# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.

struct UMAP end

"""
    UMAP(X, n_neighbors, n_components, min_dist, epochs) -> embedding

Embed the data `X` into a `n_components`-dimensional space.
# Arguments
- `X`: the dataset to embed
- `n_neighbors::Integer`: the size of the local neighborhood. Larger values
capture more global structure in the data, while smaller values capture 
more local structure
- `n_components::Integer`: the dimensionality of the embedding space
- `min_dist::AbstractFloat`: the minimum spacing of points in the 
embedding dimension. 
- `n_epochs::Integer`: the number of training epochs for embedding
optimization
"""
function UMAP(X, 
              n_neighbors::Integer,
              n_components::Integer,
              min_dist::AbstractFloat,
              epochs::Integer) 
    # argument checking
    
    # main algorithm
    fuzzy_simpl_set = local_fuzzy_simpl_set(X, n_neighbors)
    topological_repr = fuzzy_union(fuzzy_simpl_set)
    # initialize low-d embedding with spectral embedding
    X_embed = spectral_embed(topological_repr, n_components)
    # refine embedding with SGD 
    X_embed = optim_embed(topological_repr, X_embed, min_dist, n_epochs)
    return
end

"""
    local_fuzzy_simpl_set(X, n_neighbors) -> knn, knn_dists

Construct the local fuzzy simplicial sets of each point in `X` by 
finding the approximate nearest `n_neighbors`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
"""
function local_fuzzy_simpl_set(X, n_neighbors) 
    knn, knn_dists = nndescent(X, n_neighbors)
    sigma = smooth_knn_dists(knn_dists, n_neighbors)
    fuzzy_simpl_set_0 = X
    fuzzy_simpl_set_1 = # {([x, y], 0) for all x in X, y in X}
    # cont ...
end

function smooth_knn_dists(knn_dists, n_neighbors) end

function fuzzy_union(fuzzy_simpl_set) end
function spectral_embed(topological_repr, n_components) end
function optim_embed(topological_repr, X_embed, min_dist, n_epochs) end