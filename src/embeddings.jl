# initializing and optimizing embeddings

function initialize_embedding(graph::AbstractMatrix{T}, n_components, ::Val{:spectral}) where {T}
    local embed
    try
        embed = spectral_layout(graph, n_components)
        # expand
        expansion = 10 / maximum(embed)
        embed .= (embed .* expansion) .+ (1//10000) .* randn.(T)
        embed = collect(eachcol(embed))
    catch e
        @info "$e\nError encountered in spectral_layout; defaulting to random layout"
        embed = initialize_embedding(graph, n_components, Val(:random))
    end
    return embed
end

function initialize_embedding(graph::AbstractMatrix{T}, n_components, ::Val{:random}) where {T}
    return [20 .* rand(T, n_components) .- 10 for _ in 1:size(graph, 1)]
end

"""
    initialize_embedding(graph::AbstractMatrix{<:Real}, ref_embedding::AbstractMatrix{T<:AbstractFloat}) -> embedding

Initialize an embedding of points corresponding to the columns of the `graph`, by taking weighted means of
the columns of `ref_embedding`, where weights are values from the rows of the `graph`.

The resulting embedding will have shape `(size(ref_embedding, 1), length(query_inds))`, where `size(ref_embedding, 1)`
is the number of components (dimensions) of the `reference embedding`, and `length(query_inds)` is the number of 
samples in the resulting embedding. Its elements will have type T.
"""
function initialize_embedding(graph::AbstractMatrix{<:Real}, ref_embedding::AbstractMatrix{T}) where {T<:AbstractFloat}
    embed = [zeros(T, size(ref_embedding, 1)) for _ in 1:size(graph, 2)]
    for (i, col) in enumerate(eachcol(graph))
        embed[i] = vec(sum(transpose(col) .* ref_embedding, dims=2) ./ (sum(col) + eps(zero(T))))
    end
    return embed
end

"""
    spectral_layout(graph, embed_dim) -> embedding

Initialize the graph layout with spectral embedding.
"""
function spectral_layout(graph::SparseMatrixCSC{T},
                         embed_dim::Integer) where {T<:Real}
    graph_f64 = convert.(Float64, graph)
    D_ = Diagonal(dropdims(sum(graph_f64; dims=2); dims=2))
    D = inv(sqrt(D_))
    # normalized laplacian
    L = Symmetric(I - D*graph*D)

    k = embed_dim+1
    num_lanczos_vectors = max(2k+1, round(Int, sqrt(size(L, 1))))
    # get the 2nd - embed_dim+1th smallest eigenvectors
    eigenvals, eigenvecs = eigs(L; nev=k,
                                   ncv=num_lanczos_vectors,
                                   which=:SM,
                                   tol=1e-4,
                                   v0=ones(Float64, size(L, 1)),
                                   maxiter=size(L, 1)*5)
    layout = permutedims(eigenvecs[:, 2:k])::Array{Float64, 2}
    return convert.(T, layout)
end


# The optimize_embedding methods have parameters that are ::AbstractVector{<:AbstractVector{T}}.
# AbstractVector{<:AbstractVector{T}} allows arguments to be views of some other array/matrix
# rather than just vectors themselves (so we can avoid copying the model.data and instead just
# create a view to satisfy our reshaping needs). For example, calling collect(eachcol(X)) creates
# an Array of SubArrays, and SubArray is not an Array, but SubArray <: AbstractArray.

"""
    optimize_embedding(graph, embedding, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `embedding`: a vector of length (n_samples,) of vectors representing the embedded data points
- `n_epochs`::Integer: the number of training epochs for optimization
- `initial_alpha`: the initial learning rate
- `gamma`: the repulsive strength of negative samples
- `neg_sample_rate::Integer`: the number of negative samples per positive sample
"""
function optimize_embedding(graph,
                            embedding::AbstractVector{<:AbstractVector{<:AbstractFloat}},
                            n_epochs::Integer,
                            initial_alpha,
                            min_dist,
                            spread,
                            gamma,
                            neg_sample_rate,
                            _a=nothing,
                            _b=nothing)
    return optimize_embedding(
        graph,
        embedding,
        embedding,
        n_epochs,
        initial_alpha,
        min_dist,
        spread,
        gamma,
        neg_sample_rate,
        _a,
        _b,
        move_ref=true
    )
end

"""
    optimize_embedding(graph, query_embedding, ref_embedding, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate, _a=nothing, _b=nothing; move_ref=false) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.
Optimize "query" samples with respect to "reference" samples.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `query_embedding`: a vector of length (n_samples,) of vectors representing the embedded data points to be optimized
- `ref_embedding`: a vector of length (n_samples,) of vectors representing the embedded data points to optimize against
- `n_epochs`::Integer: the number of training epochs for optimization
- `initial_alpha`: the initial learning rate
- `gamma`: the repulsive strength of negative samples
- `neg_sample_rate`: the number of negative samples per positive sample
- `_a`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.
- `_b`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.

# Keyword Arguments
- `move_ref::Bool = false`: if true, also improve the embeddings in `ref_embedding`, else fix them and only improve embeddings in `query_embedding`.
"""
function optimize_embedding(graph,
                            query_embedding::AbstractVector{<:AbstractVector{T}},
                            ref_embedding::AbstractVector{<:AbstractVector{T}},
                            n_epochs::Integer,
                            initial_alpha,
                            min_dist,
                            spread,
                            gamma,
                            neg_sample_rate,
                            _a=nothing,
                            _b=nothing;
                            move_ref::Bool=false) where {T <: AbstractFloat}
    a, b = fit_ab(min_dist, spread, _a, _b)
    self_reference = query_embedding === ref_embedding

    alpha = initial_alpha
    for e in 1:n_epochs
        @inbounds for i in 1:size(graph, 2)
            for ind in nzrange(graph, i)
                j = rowvals(graph)[ind]
                p = nonzeros(graph)[ind]
                if rand() <= p
                    sdist = evaluate(SqEuclidean(), query_embedding[i], ref_embedding[j])
                    if sdist > 0
                        delta = (-2 * a * b * sdist^(b-1))/(1 + a*sdist^b)
                    else
                        delta = 0
                    end
                    @simd for d in eachindex(query_embedding[i])
                        grad = clamp(delta * (query_embedding[i][d] - ref_embedding[j][d]), -4, 4)
                        query_embedding[i][d] += alpha * grad
                        if move_ref
                            ref_embedding[j][d] -= alpha * grad
                        end
                    end

                    for _ in 1:neg_sample_rate
                        k = rand(eachindex(ref_embedding))
                        if i == k && self_reference
                            continue
                        end
                        sdist = evaluate(SqEuclidean(), query_embedding[i], ref_embedding[k])
                        if sdist > 0
                            delta = (2 * gamma * b) / ((1//1000 + sdist)*(1 + a*sdist^b))
                        else
                            delta = 0
                        end
                        @simd for d in eachindex(query_embedding[i])
                            if delta > 0
                                grad = clamp(delta * (query_embedding[i][d] - ref_embedding[k][d]), -4, 4)
                            else
                                grad = 4
                            end
                            query_embedding[i][d] += alpha * grad
                        end
                    end
                end
            end
        end
        alpha = initial_alpha*(1 - e//n_epochs)
    end
    return query_embedding
end
