
function pairwise_knn(X::AbstractMatrix, n_neighbors, metric) where {V <: AbstractVector}
    pairwise_dists = [Array{eltype(V)}(undef, length(X)) for _ in 1:length(X)]
    @inbounds for i in 1:length(X), j in 1:length(X)
        pairwise_dists[i][j] = evaluate(metric, X[i], X[j])
    end
    # get indices of closest neighbors (array of array of indices)
    knns = [p[2:n_neighbors+1] for p in sortperm.(pairwise_dists)]
    dists = [pairwise_dists[i][knns[i]] for i in 1:length(knns)]
    return hcat(knns...), hcat(dists...)    
end