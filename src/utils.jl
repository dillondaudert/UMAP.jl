
# compute all pairwise distances
# return the nearest k to each point v, other than v itself
function pairwise_knn(X::AbstractMatrix{S}, 
                      n_neighbors, 
                      metric) where {S <: AbstractFloat, V <: AbstractVector}
    num_points = size(X, 2)
    all_dists = Array{S}(undef, num_points, num_points)
    pairwise!(all_dists, metric, X)
    # all_dists is symmetric distance matrix
    knns = [partialsortperm(view(all_dists, :, i), 2:n_neighbors+1) for i in 1:num_points]
    dists = [all_dists[:, i][knns[i]] for i in eachindex(knns)]
    return hcat(knns...), hcat(dists...)
end


# combine local fuzzy simplicial sets 
function combine_fuzzy_sets(fs_set::AbstractMatrix{T}, 
                            set_op_ratio::T) where {T <: AbstractFloat}
    res = set_op_ratio .* (fs_set .+ fs_set' .- fs_set .* fs_set') .+ (one(T) - set_op_ratio) .* (fs_set .* fs_set')
    
    return res
end