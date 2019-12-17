
struct UMAPFitResult{D, M, N}
    data::D
    graph::M
    embedding::N
end

mutable struct UMAPModel
    # knn parameters
    n_neighbors
    metric
    # embedding parameters
    n_components
    init_layout
    min_dist
    spread
    set_operation_ratio
    local_connectivity
    # sgd parameters
    n_epochs
    learning_rate
    repulsion_strength
    neg_sample_rate
    _a
    _b
    # misc parameters
    verbose
end

function UMAPModel(;
    n_neighbors::Integer = 15,
    metric::Union{PreMetric, Symbol} = Euclidean(),
    n_components::Integer = 2,
    init_layout::Symbol = :spectral,
    min_dist::Real = 1//10,
    spread::Real = 1,
    set_operation_ratio::Real = 1,
    local_connectivity::Integer = 1,
    n_epochs::Integer = 300,
    learning_rate::Real = 1,
    repulsion_strength::Real = 1,
    neg_sample_rate::Integer = 5,
    a::Union{Real, Nothing} = nothing,
    b::Union{Real, Nothing} = nothing,
    verbose::Bool = false
    )
    validate_parameters(n_neighbors, metric, n_components, init_layout,
                        min_dist, spread, set_operation_ratio, local_connectivity,
                        n_epochs, learning_rate, repulsion_strength, neg_sample_rate,
                        a, b)
    return UMAPModel(n_neighbors, metric, n_components, init_layout, min_dist, spread,
                     set_operation_ratio, local_connectivity, n_epochs, learning_rate,
                     repulsion_strength, neg_sample_rate, a, b,
                     verbose)
end

function fit(model::UMAPModel, data::D) where {D <: AbstractVector}
    model._a, model._b = find_ab_params(model)
    graph = fuzzy_simplicial_set(model)
    embedding = initialize_embedding(model, graph)
    optimize_embedding!(embedding, model, graph)
    return UMAPFitResult(data, graph, embedding)
end
