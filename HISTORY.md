# Release History

## v0.3.0 (unreleased)

Performance-focused release with Float32 graph internals and Matrix embedding format.

### Performance Improvements

- **Uniform embedding initialization**: 7-14x faster, 3-10x less memory
- **Embedding optimization**: 8-10% faster due to better cache locality
- **Graph construction**: 15-28% less memory for simplicial set operations
- **Overall fit/transform**: Neutral to 7% faster, 3-20% less memory

### Breaking Changes

- **Embedding format changed**: `result.embedding` is now a `Matrix{T}` of shape `(n_dims, n_points)` instead of `Vector{Vector{T}}`
  ```julia
  # Old (v0.2)
  result.embedding[i]          # Vector for point i

  # New (v0.3)
  result.embedding[:, i]       # Column for point i
  ```

- **Graph edge weights are Float32**: `result.graph` now has `eltype` of `Float32`

### Internal Changes

- `SourceViewParams` and `SourceGlobalParams` now use fixed `Float32` fields (no longer type-parameterized)
- `MembershipFnParams.a` and `MembershipFnParams.b` are now `Float32`
- `smooth_knn_dists` returns `Float32` arrays for ρs and σs
- `SMOOTH_K_TOLERANCE` changed to `1.0f-5`

---

## v0.2.0 (2025-01-08)

Version 0.2 is a major redesign of UMAP.jl focused on generality, extensibility, and better integration with the Julia ecosystem. This release introduces breaking changes to the API and internal structure.

### New Features

- **Multi-view UMAP**: Native support for multi-modal data through `NamedTuple` inputs
  ```julia
  data = (images=X_images, text=X_text)
  result = UMAP.fit(data, ...) # advanced usage API
  ```

- **Configuration System**: Explicit configuration types for all algorithm stages
  - `NeighborParams` (with `DescentNeighbors` and `PrecomputedNeighbors` implementations)
  - `SourceViewParams` and `SourceGlobalParams` for input space control
  - `TargetParams` for embedding space configuration
  - `OptimizationParams` for SGD control
  - `UMAPConfig` bundling all parameters

- **Result Objects**: New result types that encapsulate all information
  - `UMAPResult` for fit results
  - `UMAPTransformResult` for transform results
  - Store configuration, intermediate results, and final embedding

- **Flexible Data Input**: Support for multiple input formats
  - Column-major matrices (as before)
  - Vectors of points
  - NamedTuples for multi-view data

- **Extensibility Points**:
  - Subtype `NeighborParams` for custom KNN search implementations
  - Subtype `AbstractInitialization` for custom embedding initialization
  - Multiple dispatch on manifold types for future manifold support

- **Type-safe Initialization**: Replace symbols with typed objects
  - `SpectralInitialization()` instead of `:spectral`
  - `UniformInitialization()` instead of `:random`

- **Bandwidth Parameter**: New `bandwidth` parameter for controlling smooth k-distance calculation

### Breaking Changes

#### API Changes

- **Function names**: `umap()` and `UMAP_()` replaced with `UMAP.fit()`
  ```julia
  # Old (v0.1.11)
  embedding = umap(X, 2)
  model = UMAP_(X, 2)

  # New (v0.2)
  result = UMAP.fit(X, 2)
  embedding = result.embedding
  ```

- **Export to Public**: Functions now use `public` instead of `export`
  - Access as `UMAP.fit()` and `UMAP.transform()`
  - Or use `using UMAP: fit, transform` to bring into scope

- **Transform signature**: `transform()` now takes `UMAPResult` and automatically inherits parameters
  ```julia
  # Old (v0.1.11)
  Q_embed = transform(model, Q; n_neighbors=15, min_dist=0.1, n_epochs=100)

  # New (v0.2)
  Q_result = UMAP.transform(result, Q)  # Parameters inherited
  Q_embed = Q_result.embedding
  ```

- **Result types**: `UMAP_` struct replaced with `UMAPResult`
  - Fields reorganized: `knns` and `dists` now in tuple `knns_dists`
  - Configuration stored in `config` field
  - Added `fs_sets` field for fuzzy simplicial sets

#### Parameter Changes

- **Initialization parameter**: Changed from `Symbol` to typed objects
  ```julia
  # Old
  UMAP_(X, 2; init=:spectral)

  # New
  UMAP.fit(X, 2; init=UMAP.SpectralInitialization())
  ```

- **Removed direct `a` and `b` parameters**: Now computed automatically from `min_dist` and `spread`
  - Advanced users can still customize via `MembershipFnParams` in advanced API

- **Learning rate in transform**: Default transform behavior changed
  - Now uses 30 epochs (vs 100) and learning_rate/4
  - Better defaults for transform use case

#### Internal Changes

- **Modular architecture**: Split into multiple files
  - `config.jl`, `neighbors.jl`, `simplicial_sets.jl`, `embeddings.jl`, `optimize.jl`, `fit.jl`, `transform.jl`

- **Dependency changes**:
  - Added `Accessors.jl` dependency
  - Changed from `using` to `import` for most dependencies
  - `NearestNeighborDescent` imported as `NND`

- **Embedding representation**: Internal representation changed to `Vector{Vector}` for future manifold support
  - Users typically won't notice unless accessing internal structures

### Migration Guide

**Basic Usage:**
```julia
# v0.1.11
using UMAP
embedding = umap(X, 2; n_neighbors=15, min_dist=0.1)

# v0.2
using UMAP: fit
result = fit(X, 2; n_neighbors=15, min_dist=0.1)
embedding = result.embedding
```

**Fit and Transform:**
```julia
# v0.1.11
model = UMAP_(X, 2; n_neighbors=15)
Q_embed = transform(model, Q; n_neighbors=15, n_epochs=100)

# v0.2
result = UMAP.fit(X, 2; n_neighbors=15)
Q_result = UMAP.transform(result, Q)
Q_embed = Q_result.embedding
```

**Initialization:**
```julia
# v0.1.11
result = UMAP_(X, 2; init=:random)

# v0.2
result = UMAP.fit(X, 2; init=UMAP.UniformInitialization())
```

**Accessing KNN Information:**
```julia
# v0.1.11
knns = model.knns
dists = model.dists

# v0.2
knns, dists = result.knns_dists
```

For complete migration guidance, see the [Breaking Changes documentation](docs/src/breaking_changes.md).

### Performance

- Performance characteristics remain similar to v0.1.11
- Improved type stability through explicit configuration types
- Memory layout changes may affect cache performance in some cases

### Documentation

- Comprehensive architecture documentation in landing page
- New examples demonstrating multi-view UMAP
- Detailed API reference for configuration types
- Loss function documentation with mathematical details

---

## v0.1.11 and earlier

For release notes from v0.1.x releases, see the [GitHub releases page](https://github.com/dillondaudert/UMAP.jl/releases).
