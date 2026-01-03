# UMAP.jl Source Code Architecture

This document provides a comprehensive "theory-building" mental model of the UMAP.jl codebase, documenting the key abstractions and how they fit together.

## Overview

UMAP.jl implements the Uniform Manifold Approximation and Projection (UMAP) algorithm for dimensionality reduction. The implementation is structured as a functional pipeline with clear separation of concerns across multiple stages.

## Core Pipeline

The UMAP algorithm flows through four main stages:

1. **K-Nearest Neighbors Search** → finds approximate nearest neighbors for each point
2. **Fuzzy Simplicial Set Construction** → builds a topological representation of the data
3. **Embedding Initialization** → creates initial positions in the target space
4. **Embedding Optimization** → refines the embedding via gradient descent

## File Organization

| File | Purpose |
|------|---------|
| `UMAP.jl` | Module definition and includes |
| `config.jl` | Configuration types and parameters |
| `neighbors.jl` | K-nearest neighbor search |
| `membership_fn.jl` | Membership function fitting |
| `simplicial_sets.jl` | Fuzzy simplicial set construction |
| `embeddings.jl` | Embedding initialization |
| `optimize.jl` | Embedding optimization via SGD |
| `fit.jl` | Main fitting function |
| `transform.jl` | Transform new data using fitted model |
| `utils.jl` | Utility functions for set operations and evaluation |

## Key Type Abstractions

### Configuration Types

The algorithm is parameterized by five categories of configuration:

#### 1. Neighbor Search Parameters (`NeighborParams`)
Abstract base type with two concrete implementations:

- **`DescentNeighbors{M, K}`**: Uses approximate nearest neighbor search via NearestNeighborDescent
  - Fields: `n_neighbors::Int`, `metric::M`, `kwargs::K`

- **`PrecomputedNeighbors{M}`**: Uses precomputed distances or KNN graphs
  - Fields: `n_neighbors::Int`, `dists_or_graph::M`

#### 2. Source (Input Space) Parameters

- **`SourceViewParams{T<:Real}`**: Controls fuzzy simplicial set construction per data view
  - `set_operation_ratio::T` — blend between union (1.0) and intersection (0.0)
  - `local_connectivity::T` — number of neighbors assumed locally connected
  - `bandwidth::T` — bandwidth for smooth k-distance calculation

- **`SourceGlobalParams{T<:Real}`**: Controls merging of multiple views
  - `mix_ratio::T` — ratio for weighted intersection of views

#### 3. Target (Embedding Space) Parameters

- **`TargetParams{M, D, I, P}`**: Controls the embedding space
  - `manifold::M` — target manifold (typically `_EuclideanManifold{N}`)
  - `metric::D` — distance metric (e.g., `SqEuclidean()`, `Euclidean()`)
  - `init::I` — initialization method
  - `memb_params::P` — membership function parameters

- **`MembershipFnParams{T<:Real}`**: Parameters for the target membership function
  - `min_dist::T` — minimum spacing in embedding
  - `spread::T` — effective scale of embedded points
  - `a::T`, `b::T` — curve fitting parameters (computed from min_dist/spread if not provided)

- **`AbstractInitialization`**: Base type for initialization methods
  - `SpectralInitialization` — uses spectral decomposition of graph Laplacian
  - `UniformInitialization` — random uniform initialization

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
  - `embedding::DT` — computed embedding
  - `config::C` — configuration used
  - `knns_dists::K` — k-nearest neighbors and distances
  - `fs_sets::F` — fuzzy simplicial sets (per view)
  - `graph::G` — final UMAP graph (coalesced views)

- **`UMAPTransformResult{DS, DT, K, F, G}`**: Result of transforming new data
  - Similar structure but without config (uses existing config from fit)

## Core Functions by Stage

### Stage 1: K-Nearest Neighbors Search (`neighbors.jl`)

**`knn_search(data, knn_params) -> (knns, dists)`**

Dispatches based on `knn_params` type:
- For `DescentNeighbors`: calls `nndescent()` from NearestNeighborDescent.jl
- For `PrecomputedNeighbors`: extracts from precomputed graph or constructs from distance matrix via `_knn_from_dists()`

**Multi-view support**: When `data` and `knn_params` are `NamedTuple`s, maps `knn_search` over each view.

**Transform variant**: `knn_search(data, queries, knn_params, result_knns_dists)` searches for neighbors of new queries in existing data.

Key helper:
- `_knn_from_dists(dist_mat, k; ignore_diagonal)` — extracts k-nearest neighbors from distance matrix

### Stage 2: Fuzzy Simplicial Set Construction (`simplicial_sets.jl`)

**`fuzzy_simplicial_set(knns_dists, knn_params, src_params) -> SparseMatrixCSC`**

Converts k-nearest neighbor graph to fuzzy simplicial set (weighted graph):

1. **`smooth_knn_dists(dists, k, src_params) -> (ρs, σs)`**
   - Computes smooth approximations to k-nearest neighbor distances
   - `ρs`: distances to nearest neighbors (with local connectivity interpolation)
   - `σs`: normalizing distances (found via binary search to match target perplexity)

2. **`compute_membership_strengths(knns, dists, σs, ρs) -> (rows, cols, vals)`**
   - Converts distances to membership strengths: `exp(-max(d - ρ, 0) / σ)`
   - Returns sparse matrix components

3. **`merge_local_simplicial_sets(local_fs_set, set_op_ratio) -> SparseMatrixCSC`**
   - Combines local fuzzy simplicial sets via fuzzy set union/intersection
   - `set_op_ratio` interpolates between pure union and pure intersection

**Multi-view support**:
- `fuzzy_simplicial_set` maps over multiple views when inputs are `NamedTuple`s
- `coalesce_views(view_fuzzy_sets, gbl_params)` merges multiple views using `general_simplicial_set_intersection()`

### Stage 3: Embedding Initialization (`embeddings.jl`)

**`initialize_embedding(umap_graph, tgt_params) -> embedding`**

Creates initial embedding, dispatching on manifold type and initialization method:

- **`SpectralInitialization`**:
  - `spectral_layout(graph, embed_dim)` computes normalized graph Laplacian
  - Extracts 2nd through (embed_dim+1)th smallest eigenvectors via Arpack
  - Falls back to random initialization if spectral decomposition fails

- **`UniformInitialization`**:
  - Random uniform initialization in `[-10, 10]^N`

**Transform variant**: `initialize_embedding(ref_embedding, umap_graph, tgt_params)` initializes new points as weighted averages of reference embedding.

### Stage 4: Embedding Optimization (`optimize.jl`)

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

**Transform variant**: `optimize_embedding!(embedding, ref_embedding, umap_graph, ...)` optimizes query embedding against fixed reference embedding.

**Membership function**: `fit_ab(min_dist, spread)` fits smooth curve to approximate exponential decay, returning parameters `(a, b)` used in gradient computation.

## High-Level API Functions

### `fit(data, n_components; kwargs...) -> UMAPResult`

Main entry point. Creates configuration from kwargs, then calls pipeline:

```julia
knns_dists = knn_search(data, knn_params)
fs_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_params)
umap_graph = coalesce_views(fs_sets, gbl_params)
embedding = initialize_embedding(umap_graph, tgt_params)
optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)
```

Returns `UMAPResult` containing all intermediate results.

### `transform(result::UMAPResult, queries; kwargs...) -> UMAPTransformResult`

Embeds new queries using existing model:

```julia
knns_dists = knn_search(result.data, queries, knn_params, result.knns_dists)
fs_sets = fuzzy_simplicial_set(result.data, knns_dists, knn_params, src_params)
query_graph = coalesce_views(fs_sets, gbl_params)
query_embedding = initialize_embedding(query_graph, tgt_params)
optimize_embedding!(query_embedding, result.embedding, query_graph, tgt_params, opt_params)
```

## Utility Functions (`utils.jl`)

### Fuzzy Set Operations

- **`merge_local_simplicial_sets(fs_set, set_op_ratio)`**: Merges local sets via interpolation
- **`_fuzzy_set_union(fs_set)`**: `A ∪ B = A + B - A*B`
- **`_fuzzy_set_intersection(fs_set)`**: `A ∩ B = A*B`

### Multi-View Merging

- **`general_simplicial_set_union(left, right)`**: Union of two global fuzzy sets
- **`general_simplicial_set_intersection(left, right, params)`**: Weighted intersection with mix ratio
- **`reset_local_connectivity(simplicial_set)`**: Rescales confidences to maintain local connectivity after merging

### Evaluation Metrics

- **`trustworthiness(X, X_embed, n_neighbors, metric)`**: Measures how well neighborhood structure is preserved

## Design Patterns

### 1. Multiple Dispatch for Extensibility

The codebase uses Julia's multiple dispatch extensively:
- `knn_search` dispatches on `NeighborParams` subtypes
- `initialize_embedding` dispatches on manifold and initialization types
- `_optimize_embedding!` specializes for specific manifold/metric combinations

### 2. Multi-View Architecture

The algorithm supports multiple "views" of data (different representations or modalities):
- Views are represented as `NamedTuple`s with consistent keys
- Functions map over views automatically when inputs are `NamedTuple`s
- Views are merged using `coalesce_views()` with configurable intersection strategy

### 3. Configuration via Type Parameters

Configuration structs use type parameters to:
- Enable specialization and optimization
- Avoid runtime type instabilities
- Make intent explicit (e.g., `DescentNeighbors` vs `PrecomputedNeighbors`)

### 4. Fit/Transform Split

Clear separation between:
- **Fit**: learns structure from data (combines local fuzzy sets)
- **Transform**: applies learned structure to new data (doesn't combine, just computes memberships)

Controlled by `combine` parameter in `fuzzy_simplicial_set`.

## Key Algorithms

### Smooth K-Distance Calculation

Binary search to find `σ` such that `∑ exp(-max(d - ρ, 0)/σ) ≈ log₂(k) * bandwidth`:
- Ensures perplexity approximates `k`
- `ρ` provides local connectivity offset

### Spectral Initialization

Computes normalized graph Laplacian `L = I - D^(-1/2) * G * D^(-1/2)` and extracts smallest eigenvectors (except the first, which is constant).

### Stochastic Gradient Descent

Edge-sampling SGD with:
- Probabilistic edge selection (sample edge with probability equal to edge weight)
- Negative sampling for repulsion
- Gradient clipping for stability
- Linear learning rate decay

## Constants

- **`SMOOTH_K_TOLERANCE = 1e-5`**: Tolerance for binary search in smooth k-distance and fuzzy set cardinality calculations

## Summary

This codebase implements a clean, functional pipeline for UMAP with:
- **Strong typing**: Configuration via parameterized types
- **Multiple dispatch**: Extensible algorithms via dispatch on types
- **Multi-view support**: Built-in support for multi-modal data
- **Modularity**: Clear separation of stages (neighbors → fuzzy sets → embedding → optimization)
- **Fit/transform pattern**: Separate fitting and transformation of new data

The architecture makes it straightforward to:
- Add new neighbor search methods (extend `NeighborParams`)
- Support new manifolds (extend `initialize_embedding` and `_optimize_embedding!`)
- Experiment with initialization strategies (extend `AbstractInitialization`)
- Customize fuzzy set operations (modify `merge_local_simplicial_sets`)
