# UMAP.jl Benchmark Suite Plan

## Goals

This benchmark suite measures the performance of major UMAP algorithm stages to:

1. Track performance impact of future changes
2. Identify bottlenecks in the algorithm pipeline
3. Measure both runtime and allocations separately

## Algorithm Stages to Benchmark

The UMAP algorithm has four major stages. We benchmark the UMAP-specific stages (not KNN search, which is handled by NearestNeighborDescent.jl):

### 1. Fuzzy Simplicial Set Construction (`simplicial_sets.jl`)

Converts distance information into a weighted graph representing data relationships.

**Functions to benchmark:**

| Function | Description | Parameters to vary |
|----------|-------------|-------------------|
| `fuzzy_simplicial_set` | Top-level fuzzy set construction | n_points, k, dimensions |
| `smooth_knn_dists` | Compute σ and ρ parameters | k, dimensions |
| `compute_membership_strengths` | Convert distances to probabilities | n_points, k |
| `merge_local_simplicial_sets` | Combine local sets via union/intersection | n_points, k, set_op_ratio |

### 2. Embedding Initialization (`embeddings.jl`)

Initialize the low-dimensional embedding.

**Functions to benchmark:**

| Function | Description | Parameters to vary |
|----------|-------------|-------------------|
| `initialize_embedding` (Spectral) | Spectral layout via eigenvectors | n_points, n_components, graph_density |
| `initialize_embedding` (Uniform) | Random uniform initialization | n_points, n_components |

### 3. Embedding Optimization (`optimize.jl`)

Stochastic gradient descent to optimize the embedding.

**Functions to benchmark:**

| Function | Description | Parameters to vary |
|----------|-------------|-------------------|
| `optimize_embedding!` | Full optimization loop | n_points, n_epochs, neg_sample_rate, graph_density |

### 4. Integration Benchmarks (`benchmarks.jl` - existing)

End-to-end `UMAP.fit()` to measure overall performance.

## Benchmark Configuration

### Data Sizes

| Size | n_points | Purpose |
|------|----------|---------|
| Small | 1,000 | Fast iteration, CI-friendly |
| Medium | 10,000 | Realistic workload |

### Dimensions

| Dimension | Value | Purpose |
|-----------|-------|---------|
| Low input | 10 | Low-dimensional data |
| High input | 50 | High-dimensional data |
| Low output | 2 | Typical visualization |
| High output | 10 | Higher-dimensional embedding |

### Neighbor Counts (k)

| k | Purpose |
|---|---------|
| 5 | Sparse graph |
| 15 | Default UMAP setting |
| 50 | Dense graph |

## Benchmark Structure

```
benchmark/
├── PLAN.md                     # This document
├── Project.toml                # Dependencies
├── benchmarks.jl               # Main entry point + integration benchmarks
├── utils.jl                    # Shared utilities
├── simplicial_sets_bench.jl    # Fuzzy simplicial set benchmarks
├── embeddings_bench.jl         # Embedding initialization benchmarks
└── optimize_bench.jl           # Optimization benchmarks
```

## Benchmark Tags

- `"runtime"` - Focus on execution time
- `"simplicial"` - Fuzzy simplicial set construction
- `"embeddings"` - Embedding initialization
- `"optimization"` - Embedding optimization
- `"integration"` - End-to-end fit

## Implementation Notes

1. **Pre-computed inputs**: Each stage benchmark pre-computes its inputs (e.g., KNN matrices, UMAP graphs) so we only measure the target function.

2. **GC management**: Run `GC.gc()` in setup to reduce variance from garbage collection.

3. **Sample counts**:
   - Fast operations: 20+ samples
   - Slow operations (optimization): 5-10 samples

4. **Evaluation counts**:
   - Most benchmarks: 1-2 evals per sample
   - This avoids amortizing one-time costs
