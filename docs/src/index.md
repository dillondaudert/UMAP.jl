# UMAP.jl Documentation

```@meta
CurrentModule = UMAP
```

Welcome to UMAP.jl, a pure Julia implementation of the Uniform Manifold Approximation and Projection (UMAP) algorithm for dimensionality reduction.

## What is UMAP?

**Uniform Manifold Approximation and Projection (UMAP)** is an algorithm for transforming data on one (approximated) manifold and projecting it onto another. It is used for:
- Data visualization and dimensionality reduction
- Preprocessing for machine learning
- Nonlinear dimensionality reduction, similar to t-SNE
- Theoretically motivated, computationally efficient

[McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018](https://arxiv.org/abs/1802.03426)

### Algorithm Overview

UMAP operates under three key assumptions:

1. The data is uniformly distributed on a Riemannian manifold (smooth, evenly sampled data)
2. The Riemannian metric is locally constant (we can use nearest neighbors to infer local geometry)
3. The manifold is locally connected (nearby points in high-dimensional space are also nearby on the manifold)

Given data with some representation, the algorithm projects it into a new **target** manifold through:

1. **Find nearest neighbors** - locate approximate nearest neighbors for each point
2. **Construct fuzzy simplicial sets** - build a weighted undirected graph representing the topological structure
3. **Initialize and optimize target embedding** - create initial positions and refine via stochastic gradient descent

## Quick Start

```julia
using UMAP
using Distances

# Embed data into n_components dimensions
result = UMAP.fit(data, n_components; n_neighbors=15, metric=Euclidean())
embedding = result.embedding

# Transform new data using the fitted model
transform_result = UMAP.transform(result, new_data)
new_embedding = transform_result.embedding
```

## Design and Architecture

UMAP.jl v0.2 is designed with two primary goals:
1. **Generality** - Support multiple views of input data and various target manifolds
2. **Extensibility** - Enable future functionality without major refactors

This section provides a comprehensive mental model of the UMAP.jl codebase, documenting key abstractions and how they fit together.

### Core Pipeline

The UMAP algorithm flows through four main stages:

1. **K-Nearest Neighbors Search** → finds approximate nearest neighbors for each point
2. **Fuzzy Simplicial Set Construction** → builds a topological representation of the data
3. **Embedding Initialization** → creates initial positions in the target space
4. **Embedding Optimization** → refines the embedding via gradient descent

## Key Type Abstractions

### Configuration Types

The algorithm is parameterized by five categories of configuration:

#### 1. Neighbor Search Parameters (`NeighborParams`)

Abstract base type with two concrete implementations:

- **`DescentNeighbors{M, K}`**: Uses approximate nearest neighbor search via NearestNeighborDescent
  - Fields: `n_neighbors::Int`, `metric::M`, `kwargs::K`

- **`PrecomputedNeighbors{M}`**: Uses precomputed distances or KNN graphs
  - Fields: `n_neighbors::Int`, `dists_or_graph::M`

**Extensibility**: Define new neighbor search methods by:
1. Creating a subtype `T <: NeighborParams`
2. Implementing `knn_search(data, knn_params::T)` to return nearest neighbors

Example from the implementation:
```julia
"""
    knn_search(data, knn_params::DescentNeighbors) -> (knns, dists)

Find approximate nearest neighbors using nndescent.
"""
function knn_search(data, knn_params::DescentNeighbors)
    knn_graph = NND.nndescent(data,
                              knn_params.n_neighbors,
                              knn_params.metric;
                              knn_params.kwargs...)
    return NND.knn_matrices(knn_graph)
end
```

#### 2. Source (Input Space) Parameters

- **`SourceViewParams`**: Controls fuzzy simplicial set construction per data view (all fields are `Float32`)
  - `set_operation_ratio` — blend between union (1.0) and intersection (0.0)
  - `local_connectivity` — number of neighbors assumed locally connected
  - `bandwidth` — bandwidth for smooth k-distance calculation

- **`SourceGlobalParams`**: Controls merging of multiple views (uses `Float32`)
  - `mix_ratio` — ratio for weighted intersection of views

#### 3. Target (Embedding Space) Parameters

- **`TargetParams{M, D, I, P}`**: Controls the embedding space
  - `manifold::M` — target manifold (typically `_EuclideanManifold{N}`)
  - `metric::D` — distance metric (e.g., `SqEuclidean()`, `Euclidean()`)
  - `init::I` — initialization method
  - `memb_params::P` — membership function parameters

- **`MembershipFnParams{T<:Real}`**: Parameters for the target membership function
  - `min_dist::T` — minimum spacing in embedding
  - `spread::T` — effective scale of embedded points
  - `a::Float32`, `b::Float32` — curve fitting parameters (computed from min_dist/spread if not provided)

- **`AbstractInitialization`**: Base type for initialization methods
  - `SpectralInitialization` — uses spectral decomposition of graph Laplacian
  - `UniformInitialization` — random uniform initialization

**Extensibility**: Support new target manifolds and initializations by implementing methods for new combinations of manifold and initialization types.

Example:
```julia
# Randomly initialize in Euclidean space of dimension N
# Returns a Matrix{T} of shape (N, n_points)
function initialize_embedding(
    umap_graph::AbstractMatrix{T},
    ::_EuclideanManifold{N},
    ::UniformInitialization
) where {T, N}
    return 20 .* rand(T, N, size(umap_graph, 2)) .- 10
end
```

#### 4. Optimization Parameters

- **`OptimizationParams`**: Controls stochastic gradient descent
  - `n_epochs::Int` — number of optimization epochs
  - `lr::Float64` — initial learning rate
  - `repulsion_strength::Float64` — weighting of negative samples
  - `neg_sample_rate::Int` — number of negative samples per positive sample

#### 5. Complete Configuration

- **`UMAPConfig{K, S, G, T, O}`**: Bundles all parameters together
  - Fields: `knn_params`, `src_params`, `gbl_params`, `tgt_params`, `opt_params`

### Result Types

- **`UMAPResult{DS, DT, C, K, F, G}`**: Complete result of fitting UMAP
  - `data::DS` — original data
  - `embedding::DT` — computed embedding (for Euclidean manifolds, a `Matrix{T}` of shape `(n_dims, n_points)`)
  - `config::C` — configuration used
  - `knns_dists::K` — k-nearest neighbors and distances
  - `fs_sets::F` — fuzzy simplicial sets (per view)
  - `graph::G` — final UMAP graph (coalesced views, with `Float32` edge weights)

- **`UMAPTransformResult{DS, DT, K, F, G}`**: Result of transforming new data
  - Similar structure but without config (uses existing config from fit)

## Core Functions by Stage

### Stage 1: K-Nearest Neighbors Search

**`knn_search(data, knn_params) -> (knns, dists)`**

Dispatches based on `knn_params` type:
- For `DescentNeighbors`: calls `nndescent()` from NearestNeighborDescent.jl
- For `PrecomputedNeighbors`: extracts from precomputed graph or constructs from distance matrix via `_knn_from_dists()`

**Multi-view support**: When `data` and `knn_params` are `NamedTuple`s, maps `knn_search` over each view.

**Transform variant**: `knn_search(data, queries, knn_params, result_knns_dists)` searches for neighbors of new queries in existing data.

### Stage 2: Fuzzy Simplicial Set Construction

**`fuzzy_simplicial_set(knns_dists, knn_params, src_params) -> SparseMatrixCSC`**

Converts k-nearest neighbor graph to fuzzy simplicial set (weighted graph). The elements are edges (v, w), each with a membership probability (edge weight) p(v, w).

For each view:
1. Each point has a local simplicial set, representing the local notion of distance on the manifold (defined via distances to its knn)
2. We combine these via fuzzy set union/intersection to create the global graph

The process:

1. **`smooth_knn_dists(dists, k, src_params) -> (ρs, σs)`**
   - Computes smooth approximations to k-nearest neighbor distances
   - `ρs`: distances to nearest neighbors (with local connectivity interpolation)
   - `σs`: normalizing distances (found via binary search to match target perplexity)

   Binary search finds `σ` such that `∑ exp(-max(d - ρ, 0)/σ) ≈ log₂(k) * bandwidth`, ensuring perplexity approximates `k`.

2. **`compute_membership_strengths(knns, dists, σs, ρs) -> (rows, cols, vals)`**
   - Converts distances to membership strengths: `exp(-max(d - ρ, 0) / σ)`
   - Returns sparse matrix components

3. **`merge_local_simplicial_sets(local_fs_set, set_op_ratio) -> SparseMatrixCSC`**
   - Combines local fuzzy simplicial sets via fuzzy set union/intersection
   - `set_op_ratio` interpolates between pure union and pure intersection

**Multi-view support**:
- `fuzzy_simplicial_set` maps over multiple views when inputs are `NamedTuple`s
- `coalesce_views(view_fuzzy_sets, gbl_params)` merges multiple views using `general_simplicial_set_intersection()`

The graphs of each view are coalesced into a single UMAP graph. This graph is sparse since, for each point, we only have non-zero probability to its `k` nearest neighbors, giving O(nk) entries.

### Stage 3: Embedding Initialization

**`initialize_embedding(umap_graph, tgt_params) -> embedding`**

Creates initial embedding, dispatching on manifold type and initialization method:

- **`SpectralInitialization`**:
  - `spectral_layout(graph, embed_dim)` computes normalized graph Laplacian `L = I - D^(-1/2) * G * D^(-1/2)`
  - Extracts 2nd through (embed_dim+1)th smallest eigenvectors via Arpack
  - Falls back to random initialization if spectral decomposition fails

- **`UniformInitialization`**:
  - Random uniform initialization in `[-10, 10]^N`

**Transform variant**: `initialize_embedding(ref_embedding, umap_graph, tgt_params)` initializes new points as weighted averages of reference embedding.

### Stage 4: Embedding Optimization

**`optimize_embedding!(embedding, umap_graph, tgt_params, opt_params) -> embedding`**

Optimizes embedding via stochastic gradient descent with:
- **Attractive forces**: for edges in `umap_graph`, pull points together
- **Repulsive forces**: for random negative samples, push points apart

Core optimization loop in `_optimize_embedding!()`:
- For each edge `(i, j)` with weight `p`:
  - Sample edge with probability `p`
  - Compute distance and gradient using target metric
  - Apply attractive force: gradient ∝ `-a*b / (dist * (a + dist^(-b)))`
  - Sample `neg_sample_rate` random points
  - Apply repulsive force: gradient ∝ `repulsion_strength*b / (a*dist^(b+1) + dist)`
- Gradients clipped to `[-4, 4]`
- Learning rate decays linearly across epochs

Computationally, this iterates over each non-zero edge in the graph and calculates both attractive and repulsive forces between vertices as a function of their distance in the target embedding, O(enkd), where d is the dimension of the target embedding and e is the number of epochs.

**Transform variant**: `optimize_embedding!(embedding, ref_embedding, umap_graph, ...)` optimizes query embedding against fixed reference embedding.

**Membership function**: `fit_ab(min_dist, spread)` fits smooth curve to approximate exponential decay, returning parameters `(a, b)` used in gradient computation.

## High-Level API Functions

### Basic Usage

The basic API is designed to be user-friendly with reasonable defaults:

```julia
# Embed single view into Euclidean space with dim `n_components`,
# controlling execution via keyword arguments
result = UMAP.fit(data, n_components; kwargs...)
```

Important keyword arguments:
- `n_neighbors`: Controls local neighborhood size. Larger values capture more global structure, smaller values preserve more local structure.
- `metric`: Distance metric from `Distances.jl` package (default: `Euclidean()`)
- `min_dist`: Controls minimum spacing in the embedding. Larger values spread points more evenly, smaller values preserve more local structure.

### Advanced Usage

The advanced API allows complete customization of each stage, multiple data views, etc:

```julia
result = UMAP.fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)
```

Helper functions for constructing configuration structs:
- `create_view_config(...)`
- `create_config(...)`

See the [Advanced Usage](https://dillondaudert.github.io/UMAP.jl/dev/examples/advanced_usage/) example notebook for details.

### fit(data, n_components; kwargs...) -> UMAPResult

Main entry point. Creates configuration from kwargs, then calls pipeline:

```julia
# The generic fit algorithm - works for single and named tuples of configs
function fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)
    # 1. Find (approx) nearest neighbors
    knns_dists = knn_search(data, knn_params)
    # 2. Construct the umap graph (global fuzzy simplicial set)
    fs_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_params)
    umap_graph = coalesce_views(fs_sets, gbl_params)
    # 3. Initialize the target embedding
    embedding = initialize_embedding(umap_graph, tgt_params)
    # 4. Optimize the embedding
    optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)

    config = UMAPConfig(knn_params, src_params, gbl_params, tgt_params, opt_params)
    return UMAPResult(data, embedding, config, knns_dists, fs_sets, umap_graph)
end
```

Returns `UMAPResult` containing all intermediate results.

### transform(result::UMAPResult, queries; kwargs...) -> UMAPTransformResult

Embeds new queries using existing model:

```julia
knns_dists = knn_search(result.data, queries, knn_params, result.knns_dists)
fs_sets = fuzzy_simplicial_set(result.data, knns_dists, knn_params, src_params)
query_graph = coalesce_views(fs_sets, gbl_params)
query_embedding = initialize_embedding(query_graph, tgt_params)
optimize_embedding!(query_embedding, result.embedding, query_graph, tgt_params, opt_params)
```

Note that the type of `new_data` must match the original `data` exactly. The parameterization used for `fit` is re-used where appropriate in `transform`, via the `UMAPResult` struct.

### Using Precomputed Distances

UMAP can use a precomputed distance matrix instead of finding nearest neighbors itself:

```julia
result = UMAP.fit(distances, n_components; metric=:precomputed)
```

## Utility Functions

### Fuzzy Set Operations

- **`merge_local_simplicial_sets(fs_set, set_op_ratio)`**: Merges local sets via interpolation
- **`_fuzzy_set_union(fs_set)`**: `A ∪ B = A + B - A*B`
- **`_fuzzy_set_intersection(fs_set)`**: `A ∩ B = A*B`

### Multi-View Merging

- **`general_simplicial_set_union(left, right)`**: Union of two global fuzzy sets
- **`general_simplicial_set_intersection(left, right, params)`**: Weighted intersection with mix ratio
- **`reset_local_connectivity(simplicial_set)`**: Rescales confidences to maintain local connectivity after merging

## Multi-View Data

UMAP.jl supports data with multiple "views" - different representations of the same entities. A feature vector and its label, for example, or a retail product with both text description and image.

Data with multiple views can use different notions of distance for each representation:
- Vectors in ℝⁿ with Euclidean distance (e.g., images)
- Strings with Levenshtein distance
- Categories or labels

Views are represented as `NamedTuple`s with consistent keys. Functions automatically map over views when inputs are `NamedTuple`s, and views are merged using `coalesce_views()` with configurable intersection strategy.

## Performance Considerations

### Memory Usage

The current design assumes that the UMAP graph and target embedding can fit in memory:
- The UMAP graph (sparse matrix) O(nk)
- Embedding O(nd) for an embedding dimension of d

The data itself may be larger than this, as long as the `knn_search` stage supports it properly, e.g., by precomputing nearest neighbors prior to calling into UMAP.

### Computational Complexity

In general, runtime is dominated by embedding optimization. Each stage has its own general complexity:

- `knn_search`: When using `nndescent`, empirical runtime is O(n^1.14)
- `fuzzy_simplicial_set`: Iterates over each point and its knn: O(nk)
- `optimize_embedding`: For each epoch e, we iterate over each point n
  - Each neighbor with membership probability, fixed to be log₂(k)
  - For each point, sample γ points at random for repulsive force
  - For each attractive and repulsive force, calculate distance as function of target embedding d

  Giving complexity of approximately O(e · n · log₂(k) · γ · d)
