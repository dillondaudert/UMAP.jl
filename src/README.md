# UMAP.jl Source Code Overview

This directory contains the source code for the UMAP (Uniform Manifold Approximation and Projection) Julia implementation. This is a **work-in-progress** implementation that appears to be in the middle of a major refactoring from an older codebase.

## Current Status: 🚧 **INCOMPLETE/BROKEN** 🚧

The codebase is currently in a transitional state and **will not work** as-is. The main issues are:

1. **Missing main interface**: The exported `umap` function is not implemented
2. **Missing utility functions**: Several functions referenced in the code don't exist
3. **Incomplete integration**: The new modular structure isn't fully connected

## File Overview

### Core Module Files

#### `UMAP.jl` - Main Module File
- **Purpose**: Main module definition and exports
- **Status**: 🔴 **BROKEN** - References non-existent `umap_.jl` file  
- **Exports**: `umap` (not implemented), `UMAP_` (not implemented), `transform` (not implemented)
- **Dependencies**: Arpack, Distances, LinearAlgebra, LsqFit, NearestNeighborDescent, Setfield, SparseArrays
- **Missing**: The `umap_.jl` include is referencing a file that doesn't exist in the new structure

#### `config.jl` - Configuration & Parameter Structs
- **Purpose**: Defines all parameter structs and configuration types for UMAP
- **Status**: ✅ **COMPLETE** - Appears fully implemented
- **Key Types**:
  - `NeighborParams` (abstract) with `DescentNeighbors` and `PrecomputedNeighbors`
  - `SourceViewParams` for manifold representation parameters
  - `SourceGlobalParams` for merging multiple views
  - `TargetParams` for embedding configuration
  - `MembershipFnParams` for membership function parameters
  - `OptimizationParams` for optimization settings
  - `UMAPConfig`, `UMAPResult`, `UMAPTransformResult` for results
- **Relations**: Used by all other modules for configuration

### Core Algorithm Files

#### `neighbors.jl` - Nearest Neighbor Search
- **Purpose**: Handles k-nearest neighbor search for different data types
- **Status**: 🟡 **INCOMPLETE** - Missing utility functions
- **Key Functions**:
  - `knn_search` with multiple method dispatch for different data/parameter types
  - Support for approximate neighbors via NearestNeighborDescent
  - Support for precomputed distances and KNN graphs
- **Missing**: 
  - `knn_matrices` function (referenced but not defined)
  - `HeapKNNGraph` type/constructor (from NearestNeighborDescent?)
  - `ApproximateKNNGraph` type
- **Relations**: Used by `fit.jl` and `transform.jl`

#### `simplicial_sets.jl` - Fuzzy Simplicial Set Construction
- **Purpose**: Constructs fuzzy simplicial sets from k-nearest neighbor graphs
- **Status**: 🟡 **MOSTLY COMPLETE** - Has one TODO
- **Key Functions**:
  - `fuzzy_simplicial_set` for single and multiple views
  - `coalesce_views` for merging multiple view representations
  - `smooth_knn_dists` for distance normalization
  - `compute_membership_strengths` for edge weight computation
- **TODO**: Line 158 - "set according to min k dist scale"
- **Relations**: Uses `neighbors.jl` output, feeds into `embeddings.jl`

#### `embeddings.jl` - Embedding Initialization
- **Purpose**: Initializes target embedding points in the target manifold
- **Status**: ✅ **COMPLETE** - Appears fully implemented
- **Key Features**:
  - Spectral initialization using eigenvectors
  - Uniform random initialization
  - Reference-based initialization for transforms
  - Support for different manifold types (currently only Euclidean)
- **Relations**: Uses output from `simplicial_sets.jl`, feeds into `optimize.jl`

#### `optimize.jl` - Embedding Optimization
- **Purpose**: Optimizes the embedding using gradient descent
- **Status**: ✅ **COMPLETE** - Appears fully implemented
- **Key Functions**:
  - `optimize_embedding!` for both fit and transform scenarios
  - Gradient computation for attractive and repulsive forces
  - Cross-entropy loss calculation
  - Support for different distance metrics on target manifold
- **Relations**: Uses embeddings from `embeddings.jl`, produces final result

### High-Level Interface Files

#### `fit.jl` - Main Fitting Interface
- **Purpose**: Main interface for fitting UMAP to data
- **Status**: 🟡 **INCOMPLETE** - Missing main `umap` function
- **Key Functions**:
  - `fit` with keyword arguments (complete)
  - `fit` with explicit parameters (complete)
- **Missing**: The main `umap` function that should wrap `fit`
- **Relations**: Orchestrates the entire pipeline using all other modules

#### `transform.jl` - Transform Interface
- **Purpose**: Transform new data using existing UMAP model
- **Status**: 🟡 **INCOMPLETE** - Function exists but not exported/connected
- **Key Functions**:
  - `transform` for applying fitted model to new data
- **Missing**: Integration with main interface, proper result type handling
- **Relations**: Uses similar pipeline as `fit.jl` but with reference embedding

### Utility Files

#### `utils.jl` - Utility Functions
- **Purpose**: Various utility functions for set operations and evaluation
- **Status**: ✅ **COMPLETE** - Appears fully implemented
- **Key Functions**:
  - Fuzzy set operations (union, intersection, merging)
  - Local connectivity reset functions
  - Trustworthiness evaluation for embeddings
  - Matrix utilities for sparse operations
- **Relations**: Used throughout the codebase for various operations

#### `membership_fn.jl` - Membership Function Fitting
- **Purpose**: Fits smooth membership function parameters
- **Status**: ✅ **COMPLETE** - Small but complete
- **Key Functions**:
  - `fit_ab` for fitting membership function parameters
- **Relations**: Used by `config.jl` for `MembershipFnParams`

## Dependencies & Integration

### External Dependencies
- **NearestNeighborDescent.jl**: For approximate k-NN search
- **Distances.jl**: For distance metrics
- **LsqFit.jl**: For curve fitting in membership functions
- **Arpack.jl**: For spectral initialization
- **SparseArrays.jl**: For sparse matrix operations
- **LinearAlgebra.jl**: For linear algebra operations

### Missing Dependencies
The code references several functions/types that may need to be imported or implemented:
- `knn_matrices` (possibly from NearestNeighborDescent?)
- `HeapKNNGraph` (possibly from NearestNeighborDescent?)
- `ApproximateKNNGraph` (possibly from NearestNeighborDescent?)

## What Needs to Be Done

### High Priority (Required to make it work)
1. **Implement main `umap` function** - Create the main user interface
2. **Fix missing utility functions** - Implement or properly import `knn_matrices`, `HeapKNNGraph`, etc.
3. **Remove `umap_.jl` reference** - Update `UMAP.jl` to not include the non-existent file
4. **Connect transform functionality** - Ensure `transform` is properly exported and integrated

### Medium Priority (Improvements)
1. **Complete TODO in `simplicial_sets.jl`** - Set according to min k dist scale
2. **Add more comprehensive error handling**
3. **Add input validation throughout**
4. **Optimize performance where possible**

### Low Priority (Future features)
1. **Add support for other manifold types** (beyond Euclidean)
2. **Add support for different initialization methods**
3. **Add parallel processing support**
4. **Add more distance metrics**

## Testing Status

The test suite (`test/` directory) appears to be testing an older API that expects functions like `UMAP_()` and different parameter structures. The tests will need to be updated to work with the new modular structure.

## Algorithm Overview

The UMAP algorithm implemented here follows this general flow:

1. **Neighbor Search** (`neighbors.jl`): Find k-nearest neighbors of each point
2. **Simplicial Set Construction** (`simplicial_sets.jl`): Build fuzzy simplicial sets from neighbors
3. **View Coalescing** (`simplicial_sets.jl`): Merge multiple data views if present
4. **Embedding Initialization** (`embeddings.jl`): Initialize points in target space
5. **Optimization** (`optimize.jl`): Optimize embedding via gradient descent
6. **Result Packaging** (`fit.jl`): Package results into appropriate return types

The modular design separates concerns well and should be maintainable once the missing pieces are implemented.
