#=
Utilities used by UMAP.jl
=#

# utilities to evaluate embeddings 

"""
    trustworthiness(X, X_embed, n_neighbors, metric) -> [0, 1]

Compute the trustworthiness of an embedding `X_embed` compared to `X`. 

https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
"""
function trustworthiness(X, X_embed, n_neighbors, metric)
    # FIXME: check this implementation, currently can't be used.

    n_points = length(axes(X)[end])
    if n_points != length(axes(X_embed)[end])
        error("X and X_embed must have the same dimensions; got $(size(X)), $(size(X_embed))")
    end

    # compute the exact nearest neighbors for X and X_embed

    X_nn = _pairwise_nn(X, metric)
    X_embed_nn = _pairwise_nn(X_embed, metric)

    penalty = 0.
    for i in 1:n_points
        # for each point's nearest neighors in the EMBEDDED space 
        for (j, _) in sort(collect(enumerate(X_embed_nn[i])), by=x -> x[2])[2:(n_neighbors+1)]
            # j is the point index
            # find this points' neighbor rank to i in the INPUT space 
            rank_i_j = findfirst(x -> x == j, X_nn[i])
            if isnothing(rank_i_j)
                error("findfirst returned nothing: ", j, X_nn[i])
            end
            # subtract 1 from rank_i_j as self distance is always rank 1
            penalty += max(0., (rank_i_j - 1) - n_neighbors)
        end
    end

    return 1 - (2/(n_points*n_neighbors*(2*n_points - 3*n_neighbors - 1))) * penalty

end

function _pairwise_nn(X, metric)
    # calculate the pairwise distances and return a vector of vectors 
    # nn_rank where nn_rank[i][j] returns the neighbor rank of point j to point i
    n_points = length(axes(X)[end])
    dists = zeros(n_points, n_points)

    for i in 1:n_points
        for j in 1:n_points
            dists[j, i] = metric(X[i], X[j]) 
        end
    end

    nn_ranks = [Int[] for _ in 1:n_points]
    for i in 1:n_points
        nn_ranks[i] = invperm(getindex.(sort(collect(enumerate(dists[:, i])), by=x -> x[2]), 1))
    end

    return nn_ranks
end
