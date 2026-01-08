# UMAP.jl
| **Documentation** | **Build Status** | **Test Coverage** |
|:-----------------:|:----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI](https://github.com/dillondaudert/UMAP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/dillondaudert/UMAP.jl/actions/workflows/CI.yml) | [![][codecov-img]][codecov-url] [![][coveralls-img]][coveralls-url] |

A pure Julia implementation of the [Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) dimension reduction
algorithm

> McInnes, L, Healy, J, Melville, J, *UMAP: Uniform Manifold Approximation and Projection for
> Dimension Reduction*. ArXiV 1802.03426, 2018

## Usage
```jl
result = UMAP.fit(data, n_components; n_neighbors, metric, ...) -> UMAP.UMAPResult
result.embedding
```
The `fit` function takes two arguments, `data` (either a column-major matrix or a vector of "points", e.g. vectors), `n_components` (the number of dimensions in the output embedding), and various keyword arguments. Several important ones are:
- `n_neighbors`: This controls how many neighbors around each point are considered to be part of its local neighborhood. Larger values will result in embeddings that capture more global structure, while smaller values will preserve more local structures.
- `metric`: The distance (semi-)metric to use when calculating distances between points. This can be any subtype of the `SemiMetric` type from the `Distances.jl` package, including user-defined types.
- `min_dist`: This controls the minimum spacing of points in the embedding. Larger values will cause points to be more evenly distributed, while smaller values will preserve more local structure.

`UMAP.fit` returns a `UMAPResult` struct, with the output embedding at
`result.embedding`.

### Using precomputed distances
UMAP can use a precomputed distance matrix instead of finding the nearest neighbors itself. In this case, the distance matrix is passed as `data` and the `metric` keyword argument should be `:precomputed`. Example:

```jl
result = UMAP.fit(distances, n_components; metric=:precomputed)
```

### Transforming new data

After embedding a dataset, we can transform new points into the same
embedding space via `UMAP.transform`:
```jl
result = UMAP.fit(data, n_component; <kwargs>)

transform_result = UMAP.transform(result, new_data) -> UMAP.UMAPTransformResult
transform_result.embedding
```

Note that the type of `new_data` must match the original `data`
exactly. The parameterization used for `fit` is re-used where
appropriate in `transform`, via the `UMAPResult` struct.

## Examples
The docs have more examples, e.g. 
- [MNIST](https://dillondaudert.github.io/UMAP.jl/dev/examples/mnist/)
- [Advanced Usage](https://dillondaudert.github.io/UMAP.jl/dev/examples/advanced_usage/)


## External Resources
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)
- For a great description of how UMAP works, see [this page](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) from the Python UMAP documentation
- If you're familiar with [t-SNE](https://lvdmaaten.github.io/tsne/), then [this page](https://jlmelville.github.io/uwot/umap-for-tsne.html) describes UMAP with similar vocabulary to that dimension reduction algorithm

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://dillondaudert.github.io/UMAP.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://dillondaudert.github.io/UMAP.jl/dev

[codecov-img]: https://codecov.io/gh/dillondaudert/UMAP.jl/branch/master/graph/badge.svg?token=OSn7Og8mcF
[codecov-url]: https://codecov.io/gh/dillondaudert/UMAP.jl

[coveralls-img]: https://coveralls.io/repos/github/dillondaudert/UMAP.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/dillondaudert/UMAP.jl?branch=master