# UMAP

UMAP.jl implements the Uniform Manifold Approximation and Projection algorithm for dimensionality reduction. UMAP transforms high-dimensional data into a lower-dimensional embedding while preserving both local and global structure, making it ideal for visualization and preprocessing.

## Quick Start

```julia
result = UMAP.fit(data, n_components; n_neighbors=15, metric=Euclidean())
embedding = result.embedding

# Transform new data using the fitted model
transform_result = UMAP.transform(result, new_data)
```

The `fit` function accepts column-major matrices or vectors of points. Key parameters include `n_neighbors` (controls local vs global structure preservation), `metric` (any `SemiMetric` from Distances.jl), and `min_dist` (controls spacing in the embedding).

## Documentation

For comprehensive documentation including architecture details, advanced usage with multi-view data, and examples, see:
- **Stable**: https://dillondaudert.github.io/UMAP.jl/stable
- **Development**: https://dillondaudert.github.io/UMAP.jl/dev

## Reference

McInnes, L, Healy, J, *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. ArXiv 1802.03426, 2018
