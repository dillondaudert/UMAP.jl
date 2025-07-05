# UMAP.jl Implementation Architecture Overview

This document provides a comprehensive overview of the UMAP.jl implementation architecture, covering both the intended design and current state of the codebase.

## Current Status: Implementation in Transition

⚠️ **Important**: This codebase is currently in a transitional state between an older monolithic implementation and a new modular design. The code will not run as-is and requires completion of the refactoring process.

## Design Philosophy

The UMAP.jl implementation follows a **modular, composable architecture** that separates concerns into distinct phases of the algorithm. This design enables:

1. **Flexibility**: Each component can be configured independently
2. **Extensibility**: New distance metrics, manifolds, and initialization methods can be added
3. **Multi-view Support**: Native support for combining heterogeneous data views
4. **Type Safety**: Extensive use of Julia's type system for correctness and performance

## Algorithmic Foundation

UMAP (Uniform Manifold Approximation and Projection) is built on topological data analysis principles:

1. **Manifold Approximation**: Assume data lies on a Riemannian manifold
2. **Fuzzy Simplicial Sets**: Convert local metric spaces to fuzzy topological representations
3. **Cross-Entropy Optimization**: Match high and low-dimensional topological structures

### Mathematical Framework

The algorithm constructs a **fuzzy topological representation** of the input data:

```
For dataset X = {x₁, ..., xₙ} in ℝᵐ:
1. Create local metric spaces {(X, dᵢ) | i = 1, ..., N}
2. Convert to fuzzy simplicial sets via FinSing functor
3. Combine via fuzzy union: ⋃ᵢ₌₁ᴺ FinSing((X, dᵢ))
```

The low-dimensional embedding Y in ℝᵈ undergoes similar treatment, and the two fuzzy sets are compared via **fuzzy set cross-entropy**:

```
C(A, μ, ν) = Σₐ∈A [μ(a)·log(ν(a)) + (1-μ(a))·log(1-ν(a))]
```

## Architecture Overview

The implementation is structured as a **pipeline of composable transformations**:

```
Data → KNN Search → Fuzzy Sets → Coalescing → Embedding → Optimization → Results
```

### Core Pipeline Flow

```julia
# High-level algorithm structure (from advanced_usage.jl)
knns_dists = knn_search(data, knn_params)
fuzzy_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_view_params)
umap_graph = coalesce_views(fuzzy_sets, src_global_params)
embedding = initialize_embedding(umap_graph, tgt_params)
optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)
```

## Component Architecture

### 1. Configuration System (`config.jl`)

The configuration system uses **parameter objects** to control each phase:

```julia
# Neighbor search configuration
DescentNeighbors(n_neighbors, metric, kwargs)    # Approximate search
PrecomputedNeighbors(n_neighbors, distances)     # Precomputed distances

# Source manifold representation
SourceViewParams(set_operation_ratio, local_connectivity, bandwidth)
SourceGlobalParams(mix_ratio)  # For combining multiple views

# Target embedding configuration
TargetParams(manifold, metric, init, membership_params)
MembershipFnParams(min_dist, spread, a, b)

# Optimization parameters
OptimizationParams(n_epochs, lr, repulsion_strength, neg_sample_rate)
```

**Design Principle**: Configuration objects are immutable, type-stable, and validate parameters at construction time.

### 2. Neighbor Search (`neighbors.jl`)

The neighbor search system supports multiple data types and search strategies:

```julia
# Dispatch-based design for different data/parameter combinations
knn_search(data, knn_params::DescentNeighbors)     # Approximate search
knn_search(data, knn_params::PrecomputedNeighbors) # Precomputed distances
knn_search(data::NamedTuple, knn_params::NamedTuple) # Multiple views
```

**Key Features**:
- **Approximate Search**: Uses NearestNeighborDescent.jl for scalability
- **Precomputed Support**: Handles distance matrices and KNN graphs
- **Multi-view Native**: Broadcasts over named tuples of data views
- **Transform Support**: Separate methods for fitting vs transforming

### 3. Fuzzy Simplicial Sets (`simplicial_sets.jl`)

Converts nearest neighbor graphs to fuzzy topological representations:

```julia
# Core transformation: KNN graph → Fuzzy simplicial set
fuzzy_simplicial_set(knns_dists, knn_params, src_params) → SparseMatrixCSC

# Multi-view support via broadcasting
fuzzy_simplicial_set(knns_dists::NamedTuple, ...) → NamedTuple

# View coalescing: Multiple views → Single graph
coalesce_views(view_fuzzy_sets, global_params) → SparseMatrixCSC
```

**Mathematical Operations**:
- **Distance Smoothing**: `smooth_knn_dists` normalizes distances via binary search
- **Membership Computation**: `compute_membership_strengths` calculates edge weights
- **Set Operations**: Fuzzy union/intersection for combining views

### 4. Embedding Initialization (`embeddings.jl`)

Initializes points in the target embedding space:

```julia
# Manifold-aware initialization
initialize_embedding(umap_graph, tgt_params::TargetParams)

# Multiple initialization strategies
SpectralInitialization()  # Eigenvectors of normalized Laplacian
UniformInitialization()   # Random uniform distribution

# Transform-specific initialization
initialize_embedding(ref_embedding, umap_graph, tgt_params) # Weighted average
```

**Design Features**:
- **Manifold Abstraction**: Currently supports `_EuclideanManifold{N}`, extensible to others
- **Spectral Layout**: Uses ARPACK for efficient eigenvector computation
- **Reference-based**: Transform initialization uses existing embedding as reference

### 5. Optimization (`optimize.jl`)

Performs gradient descent optimization of the cross-entropy loss:

```julia
# Main optimization loop
optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)

# Supports both fit and transform scenarios
optimize_embedding!(embedding, ref_embedding, umap_graph, ...)
```

**Optimization Strategy**:
- **Stochastic Gradient Descent**: Samples edges proportional to their strength
- **Negative Sampling**: Uniform sampling for repulsive forces
- **Adaptive Learning Rate**: Decreases linearly over epochs
- **Gradient Clipping**: Prevents instability via clipping to [-4, 4]

### 6. High-Level Interfaces

#### Simple Interface (`fit.jl`)
```julia
# High-level convenience function
fit(data, n_components; n_neighbors, metric, min_dist, ...)
```

#### Advanced Interface (Multi-view)
```julia
# Full control over all parameters
fit(data_views, knn_params, src_params, gbl_params, tgt_params, opt_params)
```

#### Transform Interface (`transform.jl`)
```julia
# Embed new data into existing embedding
transform(result::UMAPResult, queries, ...)
```

## Multi-View Architecture

A key architectural strength is **native multi-view support**:

```julia
# Multiple heterogeneous data views
data_views = (
    vectors = matrix_data,
    strings = string_data,
    graphs = graph_data
)

# Corresponding search parameters
knn_params = (
    vectors = DescentNeighbors(15, Euclidean()),
    strings = DescentNeighbors(10, Levenshtein()),
    graphs = DescentNeighbors(5, GraphDistance())
)

# View-specific source parameters
src_params = (
    vectors = SourceViewParams(1.0, 1.0, 1.0),
    strings = SourceViewParams(0.8, 1.5, 1.2),
    graphs = SourceViewParams(0.9, 2.0, 1.1)
)

# Global fusion parameters
gbl_params = SourceGlobalParams(0.5)  # 50% intersection, 50% union
```

**Multi-View Processing**:
1. **Parallel Processing**: Each view processed independently
2. **Fuzzy Set Fusion**: Views combined via fuzzy set operations
3. **Unified Optimization**: Single embedding optimized against combined graph

## Type System Design

The implementation makes extensive use of Julia's type system:

```julia
# Generic, parameterized result types
struct UMAPResult{DS, DT, C, K, F, G}
    data::DS           # Original data
    embedding::DT      # Low-dimensional embedding
    config::C          # Configuration used
    knns_dists::K      # Nearest neighbor results
    fs_sets::F         # Fuzzy simplicial sets
    graph::G           # Final UMAP graph
end

# Abstract types for extensibility
abstract type NeighborParams end
abstract type AbstractInitialization end
```

**Benefits**:
- **Type Stability**: Enables aggressive compiler optimizations
- **Extensibility**: New components can be added without modifying existing code
- **Generic Programming**: Same code works with different numeric types

## Memory and Performance Design

### Sparse Matrix Representation
- **Fuzzy Simplicial Sets**: Stored as `SparseMatrixCSC` for memory efficiency
- **Graph Operations**: Leverage Julia's sparse matrix infrastructure
- **Incremental Updates**: In-place modifications where possible

### Optimization Strategies
- **Stochastic Updates**: O(edges) rather than O(n²) per epoch
- **Gradient Clipping**: Prevents numerical instability
- **Adaptive Learning**: Annealing schedule for convergence

## Extension Points

The modular design enables several extension points:

### 1. Distance Metrics
Add new metrics by implementing `SemiMetric` interface:
```julia
struct CustomMetric <: SemiMetric end
# Implement evaluate(::CustomMetric, x, y)
```

### 2. Manifolds
Add new target manifolds:
```julia
struct SphericalManifold{N} end
# Implement distance and gradient computations
```

### 3. Initialization Methods
Add new initialization strategies:
```julia
struct CustomInitialization <: AbstractInitialization end
# Implement initialize_embedding method
```

### 4. Neighbor Search
Add new search algorithms:
```julia
struct CustomNeighbors <: NeighborParams end
# Implement knn_search methods
```

## Current Implementation Status

### ✅ Complete Components
- **Configuration System**: Fully implemented with validation
- **Simplicial Sets**: Complete with multi-view support
- **Embedding Initialization**: Spectral and uniform methods working
- **Optimization**: Full gradient descent implementation
- **Utilities**: Set operations, evaluation metrics

### 🟡 Partially Complete
- **Neighbor Search**: Core logic present, missing utility functions
- **High-Level Interfaces**: `fit` implemented, `umap` wrapper missing
- **Transform Interface**: Logic present, integration incomplete

### 🔴 Missing/Broken
- **Main Entry Point**: `umap` function not implemented
- **Module Integration**: References to non-existent `umap_.jl`
- **Utility Functions**: `knn_matrices`, `HeapKNNGraph` missing
- **Test Suite**: Tests expect old API

## Comparison with Other Implementations

### vs. Python UMAP
- **Strengths**: Type safety, native multi-view, composable design
- **Weaknesses**: Currently incomplete, smaller ecosystem

### vs. R uwot
- **Strengths**: Better multi-view support, more flexible configuration
- **Weaknesses**: Less mature, incomplete implementation

## Future Directions

### Immediate Priorities
1. **Complete Core Pipeline**: Implement missing utility functions
2. **Main Interface**: Create `umap` function wrapper
3. **Integration Testing**: End-to-end validation
4. **Documentation**: Usage examples and tutorials

### Medium-term Goals
1. **Parallel Processing**: Multi-threading for large datasets
2. **GPU Support**: CUDA kernels for optimization
3. **More Manifolds**: Hyperbolic, spherical geometries
4. **Advanced Initialization**: Learned embeddings, transfer learning

### Long-term Vision
1. **Streaming UMAP**: Online/incremental updates
2. **Hierarchical UMAP**: Multi-scale embeddings
3. **Supervised UMAP**: Incorporate label information
4. **Differentiable UMAP**: Integration with ML frameworks

## Conclusion

The UMAP.jl architecture represents a sophisticated, modular approach to manifold learning that leverages Julia's strengths in scientific computing. While currently incomplete, the design provides a solid foundation for a powerful and extensible UMAP implementation.

The key architectural principles—modularity, type safety, multi-view support, and mathematical rigor—position it well for both research and production use once the implementation is completed.
