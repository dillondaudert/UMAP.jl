#=
Utilities used by UMAP.jl
=#


@inline fit_ab(_, __, a, b) = a, b

"""
    fit_ab(min_dist, spread, _a, _b) -> a, b

Find a smooth approximation to the membership function of points embedded in ℜᵈ.
This fits a smooth curve that approximates an exponential decay offset by `min_dist`,
returning the parameters `(a, b)`.
"""
function fit_ab(min_dist, spread, ::Nothing, ::Nothing)
    ψ(d) = d >= min_dist ? exp(-(d - min_dist)/spread) : 1.
    xs = LinRange(0., spread*3, 300)
    ys = map(ψ, xs)
    @. curve(x, p) = (1. + p[1]*x^(2*p[2]))^(-1)
    result = curve_fit(curve, xs, ys, [1., 1.], lower=[0., -Inf])
    a, b = result.param
    return a, b
end


knn_search(X::AbstractVecOrMat, k, metric::Symbol; nndescent_kwargs = NamedTuple()) = knn_search(X, k, Val(metric); nndescent_kwargs=nndescent_kwargs)

# treat given matrix `X` as distance matrix
knn_search(X::AbstractVecOrMat, k, ::Val{:precomputed}; nndescent_kwargs = NamedTuple()) = _knn_from_dists(X, k)

"""
    knn_search(X, k, metric) -> knns, dists

Find the `k` nearest neighbors of each point.

`metric` may be of type:
- ::Symbol - `knn_search` is dispatched to one of the following based on the evaluation of `metric`:
- ::Val(:precomputed) - computes neighbors from `X` treated as a precomputed distance matrix.
- ::SemiMetric - computes neighbors from `X` treated as samples, using the given metric.

# Returns
- `knns`: `knns[j, i]` is the index of node i's jth nearest neighbor.
- `dists`: `dists[j, i]` is the distance of node i's jth nearest neighbor.
"""
function knn_search(X::AbstractVecOrMat,
                    k,
                    metric::SemiMetric;
                    nndescent_kwargs = NamedTuple())
    if size(X)[end] < 4096
        return knn_search(X, k, metric, Val(:pairwise))
    else
        return knn_search(X, k, metric, Val(:approximate); nndescent_kwargs=nndescent_kwargs)
    end
end

# compute all pairwise distances
# return the nearest k to each point v, other than v itself
function knn_search(X::AbstractMatrix{S},
                    k,
                    metric,
                    ::Val{:pairwise}) where {S <: Real}
    num_points = size(X, 2)
    dist_mat = Array{S}(undef, num_points, num_points)
    pairwise!(dist_mat, metric, X, dims=2)
    # all_dists is symmetric distance matrix
    return _knn_from_dists(dist_mat, k)
end

function knn_search(X::AbstractVector,
    k,
    metric,
    ::Val{:pairwise})
    num_points = length(X)
    T = result_type(metric, first(X), first(X))
    dist_mat = [i < j ? evaluate(metric, X[i], X[j]) : zero(T) for i in eachindex(X), j in eachindex(X)]
    dist_mat = Symmetric(dist_mat, :U)
    return _knn_from_dists(dist_mat, k)
end


# find the approximate k nearest neighbors using NNDescent
function knn_search(X::AbstractVecOrMat,
                    k,
                    metric,
                    ::Val{:approximate};
                    nndescent_kwargs=NamedTuple())
    knngraph = nndescent(X, k, metric; nndescent_kwargs...)
    return knn_matrices(knngraph)
end

"""
    knn_search(X, Q, k, metric, knns, dists) -> knns, dists

Given a matrix `X` and a matrix `Q`, use the given metric to compute the `k` nearest neighbors out of the
columns of `X` from the queries (columns in `Q`). 
If the matrices are large, reconstruct the approximate nearest neighbors graph of `X` using the given `knns` and `dists`,
representing indices and distances of pairwise neighbors of `X`, and use this to search for approximate nearest 
neighbors of `Q`.
If the matrices are small, search for exact nearest neighbors of `Q` by computing all pairwise distances with `X`.

`metric` may be of type:
- ::Symbol - `knn_search` is dispatched to one of the following based on the evaluation of `metric`:
- ::Val(:precomputed) - computes neighbors from `X` treated as a precomputed distance matrix.
- ::SemiMetric - computes neighbors from `X` treated as samples, using the given metric.

# Returns
- `knns`: `knns[j, i]` is the index of node i's jth nearest neighbor.
- `dists`: `dists[j, i]` is the distance of node i's jth nearest neighbor.
"""
function knn_search(X::AbstractVecOrMat, 
                    Q::AbstractVecOrMat,
                    k::Integer,
                    metric::SemiMetric,
                    knns::AbstractMatrix{<:Integer},
                    dists::AbstractMatrix{<:Real})
    if size(X)[end] < 4096
        if ndims(X) == 2
            dists = pairwise(metric, X, Q, dims=2)
        else
            dists = [evaluate(metric, X[i], Q[j]) for i in eachindex(X), j in eachindex(Q)]
        end
        return _knn_from_dists(dists, k, ignore_diagonal=false)
    else
        knngraph = HeapKNNGraph(collect(eachcol(X)), metric, knns, dists)
        return search(knngraph, collect(eachcol(Q)), k; max_candidates=8*k)
    end
end


function _knn_from_dists(dist_mat::AbstractMatrix{S}, k::Integer; ignore_diagonal=true) where {S <: Real}
    # Ignore diagonal 0 elements (which will be smallest) when distance matrix represents pairwise distances of the same set
    # If dist_mat represents distances between two different sets, diagonal elements be nontrivial
    range = (1:k) .+ ignore_diagonal
    knns  = Array{Int,2}(undef,k,size(dist_mat,2))
    dists = Array{S,2}(undef,k,size(dist_mat,2))
    for i ∈ 1:size(dist_mat, 2)
        knns[:,i]  = partialsortperm(dist_mat[ :, i], range)
        dists[:,i] = dist_mat[knns[:,i],i]
    end
    return knns, dists
end


# combine local fuzzy simplicial sets
@inline function combine_fuzzy_sets(fs_set::AbstractMatrix{T},
                                    set_op_ratio) where {T}
    return set_op_ratio .* fuzzy_set_union(fs_set) .+
           (one(T) - set_op_ratio) .* fuzzy_set_intersection(fs_set)
end

@inline function fuzzy_set_union(fs_set::AbstractMatrix)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

@inline function fuzzy_set_intersection(fs_set::AbstractMatrix)
    return fs_set .* fs_set'
end

function _fuzzy_intersection(metric1, metric2::Categorical, g::Graph, y_graph::Graph; kwargs...)
    graph = g.graph
    y = y_graph.knns.data_with_metric.data
    unknown_weight = exp(-metric2.unknown_dist)
    far_weight = exp(-metric2.far_dist)
    I, J, V = findnz(graph)
    for nz in eachindex(I,J,V)
        yi = y[I[nz]]
        yj = y[J[nz]]
        if ismissing(yi) || ismissing(yj)
            V[nz] *= unknown_weight
        elseif yi != yj
            V[nz] *= far_weight
        end
    end
    return sparse(I, J, V, size(graph)...)
end

function _fuzzy_intersection(metric1, metric2, g1::Graph, g2::Graph; mix_weight = 0.5, kwargs...)
    left = g1.graph
    right = g2.graph

    left_min = max(minimum(nonzeros(left)) / 2.0, 1.0e-8)
    right_min = min(max((1.0 - maximum(nonzeros(right))) / 2.0, 1.0e-8), 1e-4)


    result = left + right
    result_I, result_J, result_V = findnz(result)

    left_rows = rowvals(left)
    left_vals = nonzeros(left)

    right_rows = rowvals(right)
    right_vals = nonzeros(right)

    exp = mix_weight < 0.5 ? (mix_weight / (1-mix_weight)) : (1-mix_weight) / mix_weight

    for idx in eachindex(result_I, result_J, result_V)
        i = result_I[idx]
        j = result_J[idx]

        left_val = left_min
        for k in nzrange(left, j)
            if left_rows[k] == i
                left_val = left_vals[k]
            end
        end

        
        right_val = left_min
        for k in nzrange(right, j)
            if right_rows[k] == i
                right_val = right_vals[k]
            end
        end


        if left_val > left_min || right_val > right_min
            if mix_weight < 0.5
                result_V[idx] = left_val * right_val^exp
            else
                result_V[idx] = left_val^exp * right_val
            end
        end

    end
    return sparse(result_I, result_J, result_V)
end
