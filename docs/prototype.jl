# some sketches and prototypes for how the type structure
# and parameterization might be changed
using Distances

# Misc

"""Types for layout dispatch (easier than Symbols)"""
abstract type LayoutInit end
struct RandomLayout <: LayoutInit end
struct SpectralLayout <: LayoutInit end

# EXAMPLE 1 - the parameters are structs that are themselves parameterized by named tuples
struct NeighborParams{P}
    par::P
    NeighborParams(nt::NamedTuple) = new{typeof(nt)}(nt)
end
# generic helper constructors
NeighborParams(; kwargs...) = NeighborParams((; kwargs))
# add constructors for default usages
function NeighborParams(method, n_neighbors::Integer, metric::PreMetric)
    NeighborParams(method=method, n_neighbors=n_neighbors, metric=metric)
end

struct LayoutParams{P}
    par::P
    LayoutParams(nt::NamedTuple) = new{typeof(nt)}(nt)
end
LayoutParams(; kwargs...) = LayoutParams((; kwargs...))
function LayoutParams(n_components::Integer,
                      layout_init::LayoutInit,
                      min_dist::Real,
                      spread::Real,
                      set_operation_ratio::Real,
                      local_connectivity::Integer
                      )
    return LayoutParams(n_components=n_components,
                        layout_init=layout_init,
                        min_dist=min_dist,
                        spread=spread,
                        set_operation_ratio=set_operation_ratio,
                        local_connectivity=local_connectivity)
end

struct OptimizationParams{P}
    par::P
    OptimizationParams(nt::NamedTuple) = new{typeof(nt)}(nt)
end
OptimizationParams(; kwargs...) = OptimizationParams((; kwargs...))
function OptimizationParams(n_epochs::Integer, learning_rate::Real,
                            repulsion_strength::Real, neg_sample_rate::Integer)
    return OptimizationParams(n_epochs=n_epochs, learning_rate=learning_rate,
                              repulsion_strength=repulsion_strength, neg_sample_rate=neg_sample_rate)
end

# N for neighbors, L for layout, O for optimization (T for target?)
struct UMAPModel{N<:NeighborParams, L<:LayoutParams, O<:OptimizationParams}
    nn_params::N
    layout_params::L
    opt_params::O
    function UMAPModel(nn_params, layout_params, opt_params)
        new{typeof(nn_params), typeof(layout_params), typeof(opt_params)}(nn_params, layout_params, opt_params)
    end
end


# how might we distinguish between KNN search methods?
# 1 - find exact knn (?)
# 2 - find approx knn
# 3 - extract knn from distance matrix
"""
Find the approximate k-nearest neighbors
"""
function knn_search(data, nn_params::NeighborParams{P}) where {P <: NamedTuple{(:method, :n_neighbors, :metric)}}
    # this example here assumes some sort of unified API for search methods a la Neighborhood.jl
    graph = method(data, nn_params.par.n_neighbors, nn_params.par.metric)
    search(graph, data, nn_params.par.n_neighbors)
end

"""
Extract the k-nearest neighbors from a matrix of distances.
"""
function knn_search(distances, nn_param::NeighborParams{P}) where {P <: NamedTuple{(:n_neighbors)}}
    _knn_from_dists(distances, nn_param.par.n_neighbors)
end
