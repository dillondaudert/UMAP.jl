#=
Utilities used by UMAP.jl
=#

# utilities to evaluate embeddings

"""
    trustworthiness(X, X_embed, n_neighbors, metric) -> trust

Compute the trustworthiness of an embedding `X_embed` compared to `X`: the degree
to which the `n_neighbors` nearest neighbors of each embedded point were also
near neighbors in the original space. A score of 1 indicates the local structure
of `X` is perfectly retained; lower scores indicate that points were pulled into
neighborhoods they did not belong to.

`X` and `X_embed` may each be an `AbstractMatrix` (one point per column) or an
`AbstractVector` of points. `metric(x, y)` is used to compute distances in both
the original and the embedded space. `n_neighbors` must satisfy
`1 ≤ n_neighbors < n_points / 2` for the normalization to be valid.

This computes all pairwise distances in both spaces (O(n²)); it is intended for
evaluating embeddings, not for use on very large datasets.

https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html
"""
function trustworthiness(X, X_embed, n_neighbors, metric)
    n_points = length(axes(X)[end])
    if n_points != length(axes(X_embed)[end])
        error("X and X_embed must have the same number of points; got $(size(X)), $(size(X_embed))")
    end
    if !(1 <= n_neighbors < n_points / 2)
        throw(ArgumentError("n_neighbors must satisfy 1 <= n_neighbors < n_points/2 = $(n_points/2), got $n_neighbors"))
    end

    # compute the exact nearest neighbor ranks for X and X_embed
    X_nn = _pairwise_nn(X, metric)
    X_embed_nn = _pairwise_nn(X_embed, metric)

    penalty = 0.
    for i in 1:n_points, j in 1:n_points
        # for each of point i's n_neighbors nearest neighbors j in the EMBEDDED
        # space (ranks 2:(n_neighbors + 1); rank 1 is i itself), penalize by how
        # far beyond n_neighbors that point ranks in the INPUT space
        if 2 <= X_embed_nn[i][j] <= n_neighbors + 1
            # subtract 1 from the input rank as self distance is always rank 1
            penalty += max(0., (X_nn[i][j] - 1) - n_neighbors)
        end
    end

    return 1 - (2/(n_points*n_neighbors*(2*n_points - 3*n_neighbors - 1))) * penalty
end

_pointset(X::AbstractMatrix) = collect(eachcol(X))
_pointset(X) = X

function _pairwise_nn(X, metric)
    # calculate the pairwise distances and return a vector of vectors
    # nn_rank where nn_rank[i][j] returns the neighbor rank of point j to point i
    # (each point is its own rank-1 neighbor)
    points = _pointset(X)
    n_points = length(axes(points)[end])
    dists = zeros(n_points, n_points)

    for i in 1:n_points
        for j in 1:n_points
            dists[j, i] = metric(points[i], points[j])
        end
    end

    return [invperm(sortperm(dists[:, i])) for i in 1:n_points]
end
