# Internal Interface
The internal interface for `UMAP.jl`.
Some of these may be moved to the public interface as they stabilize.

## Contents
```@contents
Pages = ["internal.md"]
```

## Index
```@index
Pages = ["internal.md"]
```

## Interface

### Nearest Neighbors Interface
`UMAP.jl` allows for an extensible way to find the nearest neighbors of your data.
This is controlled by the subtypes of `NeighborParams`.

```@docs
UMAP.NeighborParams
UMAP.DescentNeighbors
UMAP.PrecomputedNeighbors
```

The neighbor parameters control dispatch for KNN search functionality, both when 
fitting and when transforming new data.

```@docs
UMAP.knn_search
UMAP.knn_search(::NamedTuple{T}, ::NamedTuple{T}) where T
UMAP.knn_search(::NamedTuple{T}, ::NamedTuple{T}, ::NamedTuple{T}, ::NamedTuple{T}) where T
UMAP.knn_search(::Any, ::UMAP.DescentNeighbors)
UMAP.knn_search(::Any, ::Any, ::UMAP.DescentNeighbors, ::Tuple)
UMAP.knn_search(::Any, ::UMAP.PrecomputedNeighbors{M}) where {M <: NearestNeighborDescent.ApproximateKNNGraph}
UMAP.knn_search(::Any, ::UMAP.PrecomputedNeighbors)
UMAP.knn_search(::Any, ::Any, ::UMAP.PrecomputedNeighbors, ::Any)
UMAP._knn_from_dists
```

### Fuzzy Simplicial Sets Interface
The second step of the UMAP algorithm involves calculating a topological
representation of the data (potentially from multiple "views"). This
behavior is controlled by the following types and functions.

```@docs
UMAP.SourceViewParams
UMAP.SourceGlobalParams
```

These config structs parameterize the fuzzy simplicial
sets created for each view, and the global fuzzy simplicial
set after combining all views (sometimes called the UMAP graph).

```@docs
UMAP.fuzzy_simplicial_set
UMAP.fuzzy_simplicial_set(::NamedTuple{T}, ::NamedTuple{T}, ::NamedTuple{T}) where T
UMAP.fuzzy_simplicial_set(::NamedTuple{T}, ::NamedTuple{T}, ::NamedTuple{T}, ::NamedTuple{T}) where T
UMAP.fuzzy_simplicial_set(::Tuple, ::Integer, ::UMAP.NeighborParams, ::UMAP.SourceViewParams, ::Bool)
```

UMAP optimizes the difference between the source fuzzy set representation
and a target representation. Therefore, if there are multiple source
views, they need to be combined. This is done by a function `coalesce_views` that combines the view simplicial sets via set union
and intersections, controlled by the `SourceGlobalParams`.

```@docs
UMAP.coalesce_views
```