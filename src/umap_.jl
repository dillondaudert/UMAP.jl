# an implementation of Uniform Manifold Approximation and Projection
# for Dimension Reduction, L. McInnes, J. Healy, J. Melville, 2018.

# NOTE: unused for now
struct UMAP_{S}
    graph::AbstractMatrix{S}
    embedding::AbstractMatrix{S}

    function UMAP_(graph::AbstractMatrix{S}, embedding::AbstractMatrix{S}) where {S<:AbstractFloat}
        issymmetric(graph) || throw(MethodError("UMAP_ constructor expected graph to be a symmetric matrix"))
        new{S}(graph, embedding)
    end
end


"""
    umap(X::AbstractMatrix[, n_components=2]; <kwargs>) -> embedding

Embed the data `X` into a `n_components`-dimensional space. `n_neighbors` controls
how many neighbors to consider as locally connected.

# Keyword Arguments
- `n_neighbors::Integer = 15`: the number of neighbors to consider as locally connected. Larger values capture more global structure in the data, while small values capture more local structure.
- `metric::SemiMetric = Euclidean()`: the metric to calculate distance in the input space
- `n_epochs::Integer = 300`: the number of training epochs for embedding optimization
- `learning_rate::AbstractFloat = 1.`: the initial learning rate during optimization
- `init::Symbol = :spectral`: how to initialize the output embedding; valid options are `:spectral` and `:random`
- `min_dist::AbstractFloat = 0.1`: the minimum spacing of points in the output embedding
- `spread::AbstractFloat = 1.0`: the effective scale of embedded points. Determines how clustered embedded points are in combination with `min_dist`.
- `set_operation_ratio::AbstractFloat = 1.0`: interpolates between fuzzy set union and fuzzy set intersection when constructing the UMAP graph (global fuzzy simplicial set). The value of this parameter should be between 1.0 and 0.0: 1.0 indicates pure fuzzy union, while 0.0 indicates pure fuzzy intersection.
- `local_connectivity::Integer = 1`: the number of nearest neighbors that should be assumed to be locally connected. The higher this value, the more connected the manifold becomes. This should not be set higher than the intrinsic dimension of the manifold.
- `repulsion_strength::AbstractFloat = 1.0`: the weighting of negative samples during the optimization process.
- `neg_sample_rate::Integer = 5`: the number of negative samples to select for each positive sample. Higher values will increase computational cost but result in slightly more accuracy.
- `a::AbstractFloat = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
- `b::AbstractFloat = nothing`: this controls the embedding. By default, this is determined automatically by `min_dist` and `spread`.
"""
function umap(args...; kwargs...)
    # this is just a convenience function for now
    return UMAP_(args...; kwargs...).embedding
end


function UMAP_(X::AbstractMatrix{S},
               n_components::Integer = 2;
               n_neighbors::Integer = 15,
               metric::SemiMetric = Euclidean(),
               n_epochs::Integer = 300,
               learning_rate::AbstractFloat = 1.,
               init::Symbol = :spectral,
               min_dist::AbstractFloat = 0.1,
               spread::AbstractFloat = 1.0,
               set_operation_ratio::AbstractFloat = 1.0,
               local_connectivity::Integer = 1,
               repulsion_strength::AbstractFloat = 1.0,
               neg_sample_rate::Integer = 5,
               a::Union{AbstractFloat, Nothing} = nothing,
               b::Union{AbstractFloat, Nothing} = nothing
               ) where {S <: AbstractFloat, V <: AbstractVector}
    # argument checking
    size(X, 2) > n_neighbors > 0|| throw(ArgumentError("size(X, 2) must be greater than n_neighbors and n_neighbors must be greater than 0"))
    size(X, 1) > n_components > 1 || throw(ArgumentError("size(X, 1) must be greater than n_components and n_components must be greater than 1"))
    min_dist > 0. || throw(ArgumentError("min_dist must be greater than 0"))
    #n_epochs > 0 || throw(ArgumentError("n_epochs must be greater than 1"))

    # main algorithm
    graph = fuzzy_simplicial_set(X, n_neighbors, metric, local_connectivity, set_operation_ratio)

    embedding = initialize_embedding(graph, n_components, Val(init))

    embedding = optimize_embedding(graph, embedding, n_epochs, learning_rate, min_dist, spread, repulsion_strength, neg_sample_rate)
    # TODO: if target variable y is passed, then construct target graph
    #       in the same manner and do a fuzzy simpl set intersection

    return UMAP_(graph, embedding)
end

"""
    fuzzy_simplicial_set(X, n_neighbors) -> graph::SparseMatrixCSC

Construct the local fuzzy simplicial sets of each point in `X` by
finding the approximate nearest `n_neighbors`, normalizing the distances
on the manifolds, and converting the metric space to a simplicial set.
"""
function fuzzy_simplicial_set(X::AbstractMatrix{V},
                              n_neighbors,
                              metric,
                              local_connectivity,
                              set_operation_ratio) where {V <: AbstractFloat}
    if size(X, 2) < 4096
        # compute all pairwise distances
        knns, dists = pairwise_knn(X, n_neighbors, metric)
    else
        knngraph = DescentGraph(X, n_neighbors, metric)
        knngraph.graph::Matrix{Tuple{S,T}} where {S<:Integer,T<:AbstractFloat}
        knns = Array{typeof(knngraph.graph[1][1])}(undef, size(knngraph.graph))
        dists = Array{typeof(knngraph.graph[1][2])}(undef, size(knngraph.graph))
        for index in eachindex(knngraph.graph)
            @inbounds knns[index] = knngraph.graph[index][1]
            @inbounds dists[index] = knngraph.graph[index][2]
        end
    end

    σs, ρs = smooth_knn_dists(dists, n_neighbors, local_connectivity)

    rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    fs_set = sparse(rows, cols, vals, size(knns, 2), size(knns, 2))
    
    res = combine_fuzzy_sets(fs_set, set_operation_ratio)
        
    return dropzeros(res)
end

"""
    smooth_knn_dists(dists, k; <kwargs>) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.

# Keyword Arguments
...
"""
function smooth_knn_dists(knn_dists::AbstractMatrix{S}, 
                          k::Integer, 
                          local_connectivity::Integer;
                          niter::Integer=64,
                          bandwidth::AbstractFloat=1.,
                          ktol = 1e-5) where {S <: Real}
    @inline minimum_nonzero(dists) = minimum(dists[dists .> 0.])
    ρs = S[minimum_nonzero(knn_dists[:, i]) for i in 1:size(knn_dists, 2)]
    σs = Array{S}(undef, size(knn_dists, 2))

    for i in 1:size(knn_dists, 2)
        @inbounds σs[i] = smooth_knn_dist(knn_dists[:, i], ρs[i], k, local_connectivity, bandwidth, niter, ktol)
    end
    return ρs, σs
end

@fastmath function smooth_knn_dist(dists::AbstractVector, ρ, k, local_connectivity, bandwidth, niter, ktol)
    target = log2(k)*bandwidth
    lo, mid, hi = 0., 1., Inf
    #psum(dists, ρ) = sum(exp.(-max.(dists .- ρ, 0.)/mid))
    for n in 1:niter
        psum = sum(exp.(-max.(dists .- ρ, 0.)./mid))
        if abs(psum - target) < ktol
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
    compute_membership_strengths(knns, dists, σ, ρ) -> rows, cols, vals

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

function initialize_embedding(graph, n_components, ::Val{:spectral})
    embed = spectral_layout(graph, n_components)
    # expand
    expansion = 10. / maximum(embed)
    @. embed = (embed*expansion) + randn(size(embed))*0.0001
    return embed
end

function initialize_embedding(graph, n_components, ::Val{:random})
    return 20. .* rand(n_components, size(graph, 1)) .- 10.
end

"""
    optimize_embedding(graph, embedding, n_epochs, initial_alpha, min_dist, spread) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low
dimensional simplicial sets using stochastic gradient descent.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `embedding`: a dense matrix of shape (n_components, n_samples)
- `neg_sample_rate::Integer`: the number of negative samples per positive sample
"""
function optimize_embedding(graph,
                            embedding,
                            n_epochs,
                            initial_alpha,
                            min_dist,
                            spread,
                            gamma,
                            neg_sample_rate)
    a, b = fit_ϕ(min_dist, spread)

    alpha = initial_alpha
    for e in 1:n_epochs

        @inbounds for i in 1:size(graph, 2)
            for ind in nzrange(graph, i)
                j = rowvals(graph)[ind]
                p = nonzeros(graph)[ind]
                if rand() <= p
                    @views sdist = evaluate(SqEuclidean(), embedding[:, i], embedding[:, j])
                    if sdist > 0.
                        delta = (-2. * a * b * sdist^(b-1))/(1. + a*sdist^b)
                    else
                        delta = 0.
                    end
                    @simd for d in 1:size(embedding, 1)
                        grad = clamp(delta * (embedding[d,i] - embedding[d,j]), -4., 4.)
                        embedding[d,i] += alpha * grad
                        embedding[d,j] -= alpha * grad
                    end

                    for _ in 1:neg_sample_rate
                        k = rand(1:size(graph, 2))
                        @views sdist = evaluate(SqEuclidean(),
                                                embedding[:, i], embedding[:, k])
                        if sdist > 0
                            delta = (2. * gamma * b) / ((0.001 + sdist)*(1. + a*sdist^b))
                        elseif i == k
                            continue
                        else
                            delta = 0.
                        end
                        @simd for d in 1:size(embedding, 1)
                            if delta > 0.
                                grad = clamp(delta * (embedding[d, i] - embedding[d, k]), -4., 4.)
                            else
                                grad = 4.
                            end
                            embedding[d, i] += alpha * grad
                        end
                    end

                end
            end
        end
        alpha = initial_alpha*(1. - e/n_epochs)
    end

    return embedding
end

"""
    fit_ϕ(min_dist, spread) -> a, b

Find a smooth approximation to the membership function of points embedded in ℜᵈ.
This fits a smooth curve that approximates an exponential decay offset by `min_dist`.
"""
function fit_ϕ(min_dist, spread)
    ψ(d) = d >= min_dist ? exp(-(d - min_dist)/spread) : 1.
    xs = LinRange(0., spread*3, 300)
    ys = map(ψ, xs)
    @. curve(x, p) = (1. + p[1]*x^(2*p[2]))^(-1)
    result = curve_fit(curve, xs, ys, [1., 1.], lower=[0., -Inf])
    a, b = result.param
    return a, b
end

"""
    spectral_layout(graph, embed_dim) -> embedding

Initialize the graph layout with spectral embedding.
"""
function spectral_layout(graph::SparseMatrixCSC{T},
                         embed_dim::Integer) where {T<:AbstractFloat}
    D_ = Diagonal(dropdims(sum(graph; dims=2); dims=2))
    D = inv(sqrt(D_))
    # normalized laplacian
    # TODO: remove sparse() when PR #30018 is merged
    L = sparse(Symmetric(I - D*graph*D))

    k = embed_dim+1
    num_lanczos_vectors = max(2k+1, round(Int, sqrt(size(L, 1))))
    local layout
    try
        # get the 2nd - embed_dim+1th smallest eigenvectors
        eigenvals, eigenvecs = eigs(L; nev=k,
                                       ncv=num_lanczos_vectors,
                                       which=:SM,
                                       tol=1e-4,
                                       v0=ones(T, size(L, 1)),
                                       maxiter=size(L, 1)*5)
        layout = permutedims(eigenvecs[:, 2:k])::Array{T, 2}
    catch e
        print("\n", e, "\n")
        print("Error occured in spectral_layout;
               falling back to random layout.\n")
        layout = 20 .* rand(T, embed_dim, size(L, 1)) .- 10
    end
    return layout
end
