# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.

struct UMAP_ end

"""
    UMAP_(X, n_neighbors, n_components, min_dist, n_epochs) -> embedding

Embed the data `X` into a `n_components`-dimensional space.
# Arguments
- `X`: the dataset to embed
- `n_neighbors::Integer`: the size of the local neighborhood. Larger values
capture more global structure in the data, while smaller values capture
more local structure
- `n_components::Integer`: the dimensionality of the embedding space
- `min_dist::AbstractFloat`: the minimum spacing of points in the
embedding dimension.
- `n_epochs::Integer`: the number of training epochs for embedding
optimization
"""
function UMAP_(X,
               n_neighbors::Integer,
               n_components::Integer,
               min_dist::AbstractFloat,
               n_epochs::Integer)
    # argument checking

    #=
    # main algorithm
    if length(X) < 4096:
        # compute all pairwise distances
    else:
        # approximate knn distances

    fs_set_graph = local_fuzzy_simpl_set(X, n_neighbors)
    # initialize low-d embedding with spectral embedding
    X_embed = spectral_embed(topological_repr, n_components)
    # refine embedding with SGD
    X_embed = optim_embed(topological_repr, X_embed, min_dist, n_epochs)

    # TODO: if target variable y is passed, then construct target graph
    #       in the same manner and do a fuzzy simpl set intersection
    =#
    return
end

"""
    local_fuzzy_simpl_set(X, n_neighbors) -> fuzzy_simpl_set::SparseMatrixCSC

Construct the local fuzzy simplicial sets of each point in `X` by
finding the approximate nearest `n_neighbors`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
"""
function local_fuzzy_simpl_set(X, n_neighbors)
    #=
    knns, dists = nndescent(X, n_neighbors)
    σ, ρ = smooth_knn_dists(dists, n_neighbors)

    rows, cols, vals = compute_membership_strengths(knns, dists, σ, ρ)
    fs_set = sparse(rows, cols, vals) # sparse matrix M[i, j] = vᵢⱼ where
                                      # vᵢⱼ is the probability that j is in the
                                      # simplicial set of i
    dropzeros(fs_set + fs_set' - fs_set .* fs_set')
    =#
    return
end

"""
    smooth_knn_dists(dists, k; <kwargs>) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.

# Arguments
...
"""
function smooth_knn_dists(knn_dists, k::AbstractFloat; n_iter::Integer=64)
    return
end

"""
    compute_membership_strengths(knns, dists, σ, ρ) -> rows, cols, strengths

Compute the membership strengths for the 1-skeleton of each fuzzy simplicial set.
"""
function compute_membership_strengths(knns, dists, σ, ρ)
    return
end

"""
    simpl_set_embedding(X, fs_set_graph, n_components, n_epochs; <kwargs>) -> embedding

Create an embedding by minimizing the fuzzy set cross entropy between the
fuzzy simplicial set 1-skeletons of the data in high and low dimensional
spaces.
"""
function simpl_set_embedding(X, fs_set_graph::SparseMatrixCSC, n_components, n_epochs;
                             init::Symbol=:spectral)
    return
end

"""
Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low
dimensional simplicial sets using stochastic gradient descent.
"""
function optimize_embedding(head_embedding, tail_embedding, alpha=1.0)
    # fit ϕ, ψ
    #
    return
end

"""
    fit_ϕ(min_dist, spread) -> a, b

Find a smooth approximation to the membership function of points embedded in ℜᵈ.
This fits a smooth curve that approximates an exponential decay offset by `min_dist`.
"""
function fit_ϕ(min_dist, spread)
    ψ(d) = d > 0. ? exp(-(d - min_dist)/spread) : 1.
    xs = LinRange(0., spread*3, 300)
    ys = map(ψ, xs)
    @. curve(x, p) = (1. + p[1]*x^(2*p[2]))^(-1)
    result = curve_fit(curve, xs, ys, [1., 1.])
    a, b = result.param
    return a, b
end

"""
    spectral_layout(graph, embed_dim) -> embedding

Initialize the graph layout with spectral embedding.
"""
function spectral_layout(graph::SparseMatrixCSC, embed_dim)
    D_ = Diagonal(dropdims(sum(graph; dims=2); dims=2))
    D = inv(sqrt(D_))
    # normalized laplacian
    # TODO: remove sparse() when PR #30018 is merged
    L = sparse(Symmetric(I - D*graph*D))

    k = embed_dim+1
    num_lanczos_vectors = max(2k+1, round(Int, sqrt(size(L)[1])))
    local layout
    try
        # get the 2nd - embed_dim+1th smallest eigenvectors
        eigenvals, eigenvecs = eigs(L; nev=k,
                                       ncv=num_lanczos_vectors,
                                       which=:SM,
                                       tol=1e-4,
                                       v0=ones(size(L)[1]),
                                       maxiter=size(L)[1]*5)
        layout = eigenvecs[:, 2:k]
    catch e
        print(e)
        print("Error occured in spectral_layout;
               falling back to random layout.")
        layout = 20 .* rand(Float64, size(L)[1], embed_dim) .- 10
    end
    return layout
end
