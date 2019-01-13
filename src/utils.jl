
"""
    knn_search(X, k, metric) -> knns, dists

Find the `k` nearest neighbors of each point in `X` by `metric`.
"""
function knn_search(X::AbstractMatrix,
                    k::Integer,
                    metric::SemiMetric)
    if size(X, 2) < 4096
        return knn_search(X, k, metric, Val(:pairwise))
    else
        return knn_search(X, k, metric, Val(:approximate))
    end
end

function knn_search(X::AbstractMatrix{S}, 
                    k, 
                    metric, 
                    ::Val{:approximate}) where {S <: AbstractFloat}
    knngraph = DescentGraph(X, k, metric)
    knns = Array{Int}(undef, size(knngraph.graph))
    dists = Array{S}(undef, size(knngraph.graph))
    for index in eachindex(knngraph.graph)
        @inbounds knns[index] = knngraph.graph[index][1]
        @inbounds dists[index] = knngraph.graph[index][2]
    end

    return knns, dists    
end

# compute all pairwise distances
# return the nearest k to each point v, other than v itself
function knn_search(X::AbstractMatrix{S}, 
                    k, 
                    metric,
                    ::Val{:pairwise}) where {S <: AbstractFloat}
    num_points = size(X, 2)
    all_dists = Array{S}(undef, num_points, num_points)
    pairwise!(all_dists, metric, X)
    # all_dists is symmetric distance matrix
    knns_ = [partialsortperm(view(all_dists, :, i), 2:k+1) for i in 1:num_points]
    dists_ = [all_dists[:, i][knns_[i]] for i in eachindex(knns_)]
    knns = hcat(knns_...)::Matrix{Int}
    dists = hcat(dists_...)::Matrix{S}
    return knns, dists
end


# combine local fuzzy simplicial sets 
@inline function combine_fuzzy_sets(fs_set::AbstractMatrix{T}, 
                            set_op_ratio::T) where {T <: AbstractFloat}
    return set_op_ratio .* fuzzy_set_union(fs_set) .+ 
           (one(T) - set_op_ratio) .* fuzzy_set_intersection(fs_set)
end

@inline function fuzzy_set_union(fs_set::AbstractMatrix)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

@inline function fuzzy_set_intersection(fs_set::AbstractMatrix)
    return fs_set .* fs_set'
end