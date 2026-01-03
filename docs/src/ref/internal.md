# Internal Interface
The internal interface for `UMAP.jl`.
Some of these may be moved to the public interface as they stabilize.

## Contents
```@contents
Pages = ["internal.md"]
```

## Interface

### Nearest Neighbors Interface
`UMAP.jl` allows for an extensible way to find the nearest neighbors of your data.
This is controlled by the subtypes of `NeighborParams`.

```@autodocs
Modules = [UMAP]
Pages = ["neighbors.jl"]
Order = [:type]
Public = false
```

The neighbor parameters control dispatch for KNN search functionality, both when fitting and when transforming new data.

```@autodocs
Modules = [UMAP]
Pages = ["neighbors.jl"]
Order = [:function]
Public = false
```

### Fuzzy Simplicial Sets Interface
The second step of the UMAP algorithm involves calculating a topological
representation of the data (potentially from multiple "views"). This
behavior is controlled by the following types and functions.

```@autodocs
Modules = [UMAP]
Pages = ["simplicial_sets.jl"]
Order = [:constant, :type]
Public = false
```

These config structs parameterize the fuzzy simplicial
sets created for each view, and the global fuzzy simplicial
set after combining all views (sometimes called the UMAP graph).

UMAP optimizes the difference between the source fuzzy set representation
and a target representation. Therefore, if there are multiple source
views, they need to be combined. This is done by a function `coalesce_views` that combines the view simplicial sets via set union
and intersections, controlled by the `SourceGlobalParams`.

```@autodocs
Modules = [UMAP]
Pages = ["simplicial_sets.jl"]
Order = [:function]
Public = false
```

## Remaining
```@autodocs
Modules = [UMAP]
Pages = ["config.jl", "embeddings.jl",
         "membership_fn.jl", "optimize.jl",
         "utils.jl"]
Public = false
```

## Index
```@index
Pages = ["internal.md"]
```