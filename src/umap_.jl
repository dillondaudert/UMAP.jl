# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.


Base.@kwdef struct Params{M, T, NT <: NamedTuple}
    metric::M
    n_neighbors::Int = 15
    local_connectivity::Int = 1
    set_operation_ratio::T = 1
    nndescent_kwargs::NT = NamedTuple()
end

struct DataView{P<:Params, D}
    params::P
    data::D
end

DataView(X; kwargs...) = DataView(Params(; kwargs...), X)

params(d::DataView) = d.params
n_points(d::DataView) = size(d.data)[end]

Base.@kwdef struct Categorical{T}
    far_dist::T = 5.0
    unknown_dist::T = 1.0
end

struct KNNs{DD<:DataView,K,D}
    data_with_metric::DD
    knns::K
    dists::D
end

# For categorical data, we don't compute KNNs
KNNs(d::DataView{<:Params{<:Categorical}}) = KNNs(d, nothing, nothing)

function KNNs(d::DataView)
    knns, dists = knn_search(d.data, params(d).n_neighbors,  params(d).metric; nndescent_kwargs =  params(d).nndescent_kwargs)
    return KNNs(d, knns, dists)
end

params(k::KNNs) = params(k.data_with_metric)
n_points(k::KNNs) = n_points(k.data_with_metric)

struct Graph{K,G}
    knns::K
    graph::G
end

params(g::Graph) = params(g.knns)
n_points(g::Graph) = n_points(g.knns)

# For categorical data, we don't compute a fuzzy graph
Graph(knns::KNNs{<:DataView{<:Params{<:Categorical}}}) = Graph(knns, nothing)

function Graph(knns::KNNs)
    graph = fuzzy_simplicial_set(knns.knns, knns.dists, params(knns).n_neighbors, n_points(knns), params(knns).local_connectivity, params(knns).set_operation_ratio)
    return Graph(knns, graph)
end

Graph(d::DataView) = Graph(KNNs(d))
Graph(g::Graph) = g

metric(x) = params(x).metric

fuzzy_intersection(g1::Graph, g2::Graph; kwargs...) = Graph(nothing, _fuzzy_intersection(metric(g1), metric(g2), g1, g2; kwargs...))


struct Embedding{G<:Graph, E, NT <: NamedTuple}
    graph::G
    embedding::E
    embedding_params::NT
end

const DEFAULT_EMBEDDING_PARAMS = Ref((; n_components=2, n_epochs = 300, learning_rate = 1, init = :spectral, min_dist = 1//10, spread=1, repulsion_strength = 1, neg_sample_rate = 5, a = nothing, b = nothing ))

function Embedding(g::Graph; kwargs...)
    embedding_params = merge(DEFAULT_EMBEDDING_PARAMS[], kwargs)

    init_embedding = initialize_embedding(g.graph, embedding_params.n_components, Val(embedding_params.init))

    embedding = optimize_embedding(g.graph, init_embedding, init_embedding, embedding_params.n_epochs, embedding_params.learning_rate, embedding_params.min_dist, embedding_params.spread, embedding_params.repulsion_strength, embedding_params.neg_sample_rate, move_ref=true)

    return Embedding(g, reduce(hcat, embedding), embedding_params)
end

function Embedding(g1::Graph, g2::Graph; mix_weight=0.5, kwargs...)
    return Embedding(fuzzy_intersection(g1, g2); mix_weight=mix_weight); kwargs...)
end

Embedding(args...; kwargs...) = Embedding(map(Graph, args); kwargs...)


const SMOOTH_K_TOLERANCE = 1e-5


# """
#     transform(model::UMAP_, Q::AbstractVecOrMat; <kwargs>) -> embedding

# Use the given model to embed new points into an existing embedding. `Q` is a matrix of some number of points (columns)
# or a vector of data points in the same space as `model.data`. The returned embedding is the embedding of these points in n-dimensional space, where
# n is the dimensionality of `model.embedding`. This embedding is created by finding neighbors of `Q` in `model.embedding`
# and optimizing cross entropy according to membership strengths according to these neighbors.

# # Keyword Arguments
# - `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
# - `metric::{SemiMetric, Symbol} = Euclidean()`: the metric to calculate distance in the input space. It is also possible to pass `metric = :precomputed` to treat `X` like a precomputed distance matrix.
# - `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
# - `learning_rate::Real = 1`: the initial learning rate during optimization
# - `init::Symbol = :spectral`: how to initialize the output embedding; valid options are `:spectral` and `:random`
# - `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
# - `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
# - `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
# - `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
# - `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
# - `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
# - `a = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
# - `b = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
# """
# function transform(model::UMAP_,
#                    Q::AbstractVecOrMat;
#                    n_neighbors::Integer = 15,
#                    metric::Union{SemiMetric, Symbol} = Euclidean(),
#                    n_epochs::Integer = 100,
#                    learning_rate::Real = 1,
#                    min_dist::Real = 1//10,
#                    spread::Real = 1,
#                    set_operation_ratio::Real = 1,
#                    local_connectivity::Integer = 1,
#                    repulsion_strength::Real = 1,
#                    neg_sample_rate::Integer = 5,
#                    a::Union{Real, Nothing} = nothing,
#                    b::Union{Real, Nothing} = nothing
#                    )
#     # argument checking
#     size(Q)[end] > n_neighbors > 0                              || throw(ArgumentError("`size(Q)[end]` must be greater than n_neighbors and n_neighbors must be greater than 0"))
#     learning_rate > 0                                           || throw(ArgumentError("learning_rate must be greater than 0"))
#     min_dist > 0                                                || throw(ArgumentError("min_dist must be greater than 0"))
#     0 ≤ set_operation_ratio ≤ 1                                 || throw(ArgumentError("set_operation_ratio must lie in [0, 1]"))
#     local_connectivity > 0                                      || throw(ArgumentError("local_connectivity must be greater than 0"))
#     size(model.data)[end] == size(model.embedding, 2)           || throw(ArgumentError("model.data must have same number of columns or data points as model.embedding"))
#     ndims(model.data) == 1 || size(model.data, 1) == size(Q, 1) || throw(ArgumentError("size(model.data, 1) must equal size(Q, 1)"))
    

#     n_epochs = max(0, n_epochs)
#     # main algorithm
#     knns, dists = knn_search(model.data, Q, n_neighbors, metric, model.knns, model.dists)
#     graph = fuzzy_simplicial_set(knns, dists, n_neighbors, size(model.data)[end], local_connectivity, set_operation_ratio, false)

#     embedding = initialize_embedding(graph, model.embedding)
#     ref_embedding = collect(eachcol(model.embedding))
#     embedding = optimize_embedding(graph, embedding, ref_embedding, n_epochs, learning_rate, min_dist, spread, repulsion_strength, neg_sample_rate, a, b, move_ref=false)

#     return reduce(hcat, embedding)
# end


"""
    fuzzy_simplicial_set(knns, dists, n_neighbors, n_points, local_connectivity, set_op_ratio, apply_fuzzy_combine=true) -> membership_graph::SparseMatrixCSC, 

Construct the local fuzzy simplicial sets of each point represented by its distances
to its `n_neighbors` nearest neighbors, stored in `knns` and `dists`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
`n_points` indicates the total number of points of the original data, while `knns` contains
indices of some subset of those points (ie some subset of 1:`n_points`). If `knns` represents
neighbors of the elements of some set with itself, then `knns` should have `n_points` number of
columns. Otherwise, these two values may be inequivalent.
If `apply_fuzzy_combine` is true, use intersections and unions to combine
fuzzy sets of neighbors (default true).

The returned graph will have size (`n_points`, size(knns, 2)).
"""
function fuzzy_simplicial_set(knns,
                              dists,
                              n_neighbors,
                              n_points,
                              local_connectivity,
                              set_operation_ratio,
                              apply_fuzzy_combine=true)

    σs, ρs = smooth_knn_dists(dists, n_neighbors, local_connectivity)

    rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    fs_set = sparse(rows, cols, vals, n_points, size(knns, 2))

    if apply_fuzzy_combine
        res = combine_fuzzy_sets(fs_set, set_operation_ratio)
        return dropzeros(res)
    else
        return dropzeros(fs_set)
    end
end

"""
    smooth_knn_dists(dists, k, local_connectivity; <kwargs>) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.
"""
function smooth_knn_dists(knn_dists::AbstractMatrix{S},
                          k::Real,
                          local_connectivity::Real;
                          niter::Integer=64,
                          bandwidth::Real=1) where {S <: Real}

    nonzero_dists(dists) = @view dists[dists .> 0.]
    ρs = zeros(S, size(knn_dists, 2))
    σs = Array{S}(undef, size(knn_dists, 2))
    for i in 1:size(knn_dists, 2)
        nz_dists = nonzero_dists(knn_dists[:, i])
        if length(nz_dists) >= local_connectivity
            index = floor(Int, local_connectivity)
            interpolation = local_connectivity - index
            if index > 0
                ρs[i] = nz_dists[index]
                if interpolation > SMOOTH_K_TOLERANCE
                    ρs[i] += interpolation * (nz_dists[index+1] - nz_dists[index])
                end
            else
                ρs[i] = interpolation * nz_dists[1]
            end
        elseif length(nz_dists) > 0
            ρs[i] = maximum(nz_dists)
        end
        @inbounds σs[i] = smooth_knn_dist(knn_dists[:, i], ρs[i], k, bandwidth, niter)
    end

    return ρs, σs
end

# calculate sigma for an individual point
@fastmath function smooth_knn_dist(dists::AbstractVector, ρ, k, bandwidth, niter)
    target = log2(k)*bandwidth
    lo, mid, hi = 0., 1., Inf
    for n in 1:niter
        psum = sum(exp.(-max.(dists .- ρ, 0.)./mid))
        if abs(psum - target) < SMOOTH_K_TOLERANCE
            break
        end
        if psum > target
            hi = mid
            mid = (lo + hi)/2.
        else
            lo = mid
            if hi == Inf
                mid *= 2.
            else
                mid = (lo + hi) / 2.
            end
        end
    end
    # TODO: set according to min k dist scale
    return mid
end

"""
    compute_membership_strengths(knns, dists, σs, ρs) -> rows, cols, vals

Compute the membership strengths for the 1-skeleton of each fuzzy simplicial set.
"""
function compute_membership_strengths(knns::AbstractMatrix{S},
                                      dists::AbstractMatrix{T},
                                      ρs::Vector{T},
                                      σs::Vector{T}) where {S <: Integer, T}
    # set dists[i, j]
    rows = sizehint!(S[], length(knns))
    cols = sizehint!(S[], length(knns))
    vals = sizehint!(T[], length(knns))
    for i in 1:size(knns, 2), j in 1:size(knns, 1)
        @inbounds if i == knns[j, i] # dist to self
            d = 0.
        else
            @inbounds d = exp(-max(dists[j, i] - ρs[i], 0.)/σs[i])
        end
        append!(cols, i)
        append!(rows, knns[j, i])
        append!(vals, d)
    end
    return rows, cols, vals
end
