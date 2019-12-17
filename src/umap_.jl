# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.

# NOTE: unused for now
struct UMAP_{S <: Real, M <: AbstractMatrix{S}, N <: AbstractMatrix{S}}
    graph::M
    embedding::N

    function UMAP_{S, M, N}(graph, embedding) where {S<:Real,
                                                     M<:AbstractMatrix{S},
                                                     N<:AbstractMatrix{S}}
        issymmetric(graph) || isapprox(graph, graph') || error("UMAP_ constructor expected graph to be a symmetric matrix")
        new(graph, embedding)
    end
end
function UMAP_(graph::M, embedding::N) where {S <: Real,
                                              M <: AbstractMatrix{S},
                                              N <: AbstractMatrix{S}}
    return UMAP_{S, M, N}(graph, embedding)
end

const SMOOTH_K_TOLERANCE = 1e-5


"""
    umap(X::AbstractMatrix[, n_components=2]; <kwargs>) -> embedding

Embed the data `X` into a `n_components`-dimensional space. `n_neighbors` controls
how many neighbors to consider as locally connected.

# Keyword Arguments
- `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
- `metric::{SemiMetric, Symbol} = Euclidean()`: the metric to calculate distance in the input space. It is also possible to pass `metric = :precomputed` to treat `X` like a precomputed distance matrix.
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::Real = 1`: the initial learning rate during optimization
- `init::Symbol = :spectral`: how to initialize the output embedding; valid options are `:spectral` and `:random`
- `min_dist::Real = 0.1`: the minimum spacing of points in the output embedding
- `spread::Real = 1`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::Real = 1`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::Real = 1`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `a = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `b = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
"""
function umap(args...; kwargs...)
    # this is just a convenience function for now
    return UMAP_(args...; kwargs...).embedding
end


function UMAP_(X::AbstractMatrix{S},
               n_components::Integer = 2;
               n_neighbors::Integer = 15,
               metric::Union{SemiMetric, Symbol} = Euclidean(),
               n_epochs::Integer = 300,
               learning_rate::Real = 1,
               init::Symbol = :spectral,
               min_dist::Real = 1//10,
               spread::Real = 1,
               set_operation_ratio::Real = 1,
               local_connectivity::Integer = 1,
               repulsion_strength::Real = 1,
               neg_sample_rate::Integer = 5,
               a::Union{Real, Nothing} = nothing,
               b::Union{Real, Nothing} = nothing
               ) where {S<:Real}
    # argument checking
    size(X, 2) > n_neighbors > 0|| throw(ArgumentError("size(X, 2) must be greater than n_neighbors and n_neighbors must be greater than 0"))
    size(X, 1) > n_components > 1 || throw(ArgumentError("size(X, 1) must be greater than n_components and n_components must be greater than 1"))
    n_epochs > 0 || throw(ArgumentError("n_epochs must be greater than 0"))
    learning_rate > 0 || throw(ArgumentError("learning_rate must be greater than 0"))
    min_dist > 0 || throw(ArgumentError("min_dist must be greater than 0"))
    0 ≤ set_operation_ratio ≤ 1 || throw(ArgumentError("set_operation_ratio must lie in [0, 1]"))
    local_connectivity > 0 || throw(ArgumentError("local_connectivity must be greater than 0"))



    # main algorithm
    graph = fuzzy_simplicial_set(X, n_neighbors, metric, local_connectivity, set_operation_ratio)

    embedding = initialize_embedding(graph, n_components, Val(init))

    embedding = optimize_embedding(graph, embedding, n_epochs, learning_rate, min_dist, spread, repulsion_strength, neg_sample_rate)
    # TODO: if target variable y is passed, then construct target graph
    #       in the same manner and do a fuzzy simpl set intersection

    return UMAP_(graph, embedding)
end

"""
    fuzzy_simplicial_set(X, n_neighbors, metric, local_connectivity, set_op_ratio) -> graph::SparseMatrixCSC

Construct the local fuzzy simplicial sets of each point in `X` by
finding the approximate nearest `n_neighbors`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
"""
function fuzzy_simplicial_set(X,
                              n_neighbors,
                              metric,
                              local_connectivity,
                              set_operation_ratio)

    knns, dists = knn_search(X, n_neighbors, metric)

    σs, ρs = smooth_knn_dists(dists, n_neighbors, local_connectivity)

    rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    fs_set = sparse(rows, cols, vals, size(knns, 2), size(knns, 2))

    res = combine_fuzzy_sets(fs_set, set_operation_ratio)

    return dropzeros(res)
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

function initialize_embedding(graph::AbstractMatrix{T}, n_components, ::Val{:spectral}) where {T}
    local embed
    try
        embed = spectral_layout(graph, n_components)
        # expand
        expansion = 10 / maximum(embed)
        embed .= (embed .* expansion) .+ (1//10000) .* randn.(T)
    catch e
        print("Error encountered in spectral_layout; defaulting to random layout\n")
        embed = initialize_embedding(graph, n_components, Val(:random))
    end
    return embed
end

function initialize_embedding(graph::AbstractMatrix{T}, n_components, ::Val{:random}) where {T}
    return 20 .* rand(T, n_components, size(graph, 1)) .- 10
end

"""
    optimize_embedding(graph, embedding, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `embedding`: a dense matrix of shape (n_components, n_samples)
- `n_epochs`: the number of training epochs for optimization
- `initial_alpha`: the initial learning rate
- `gamma`: the repulsive strength of negative samples
- `neg_sample_rate::Integer`: the number of negative samples per positive sample
"""
function optimize_embedding(graph,
                            embedding,
                            n_epochs,
                            initial_alpha,
                            min_dist,
                            spread,
                            gamma,
                            neg_sample_rate,
                            _a=nothing,
                            _b=nothing)
    a, b = fit_ab(min_dist, spread, _a, _b)

    alpha = initial_alpha
    for e in 1:n_epochs

        @inbounds for i in 1:size(graph, 2)
            for ind in nzrange(graph, i)
                j = rowvals(graph)[ind]
                p = nonzeros(graph)[ind]
                if rand() <= p
                    @views sdist = evaluate(SqEuclidean(), embedding[:, i], embedding[:, j])
                    if sdist > 0
                        delta = (-2 * a * b * sdist^(b-1))/(1 + a*sdist^b)
                    else
                        delta = 0
                    end
                    @simd for d in 1:size(embedding, 1)
                        grad = clamp(delta * (embedding[d,i] - embedding[d,j]), -4, 4)
                        embedding[d,i] += alpha * grad
                        embedding[d,j] -= alpha * grad
                    end

                    for _ in 1:neg_sample_rate
                        k = rand(1:size(graph, 2))
                        @views sdist = evaluate(SqEuclidean(),
                                                embedding[:, i], embedding[:, k])
                        if sdist > 0
                            delta = (2 * gamma * b) / ((1//1000 + sdist)*(1 + a*sdist^b))
                        elseif i == k
                            continue
                        else
                            delta = 0
                        end
                        @simd for d in 1:size(embedding, 1)
                            if delta > 0
                                grad = clamp(delta * (embedding[d, i] - embedding[d, k]), -4, 4)
                            else
                                grad = 4
                            end
                            embedding[d, i] += alpha * grad
                        end
                    end

                end
            end
        end
        alpha = initial_alpha*(1 - e//n_epochs)
    end

    return embedding
end

"""
    spectral_layout(graph, embed_dim) -> embedding

Initialize the graph layout with spectral embedding.
"""
function spectral_layout(graph::SparseMatrixCSC{T},
                         embed_dim::Integer) where {T<:Real}
    D_ = Diagonal(dropdims(sum(graph; dims=2); dims=2))
    D = inv(sqrt(D_))
    # normalized laplacian
    # TODO: remove sparse() when PR #30018 is merged
    L = sparse(Symmetric(I - D*graph*D))

    k = embed_dim+1
    num_lanczos_vectors = max(2k+1, round(Int, sqrt(size(L, 1))))
    # get the 2nd - embed_dim+1th smallest eigenvectors
    eigenvals, eigenvecs = eigs(L; nev=k,
                                   ncv=num_lanczos_vectors,
                                   which=:SR,
                                   tol=1e-4,
                                   v0=ones(T, size(L, 1)),
                                   maxiter=size(L, 1)*5)
    layout = permutedims(eigenvecs[:, 2:k])::Array{T, 2}
    return layout
end
