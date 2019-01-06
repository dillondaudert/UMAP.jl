
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