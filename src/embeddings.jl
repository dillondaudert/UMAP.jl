# initializing and optimizing embeddings

function initialize_embedding(graph::AbstractMatrix{T}, n_components, ::Val{:spectral}) where {T}
    local embed
    try
        embed = spectral_layout(graph, n_components)
        # expand
        expansion = 10 / maximum(embed)
        embed .= (embed .* expansion) .+ (1//10000) .* randn.(T)
    catch e
        @info "$e\nError encountered in spectral_layout; defaulting to random layout"
        embed = initialize_embedding(graph, n_components, Val(:random))
    end
    return embed
end

function initialize_embedding(graph::AbstractMatrix{T}, n_components, ::Val{:random}) where {T}
    20 .* rand(T, (n_components, size(graph, 1))) .- 10
end

"""
    initialize_embedding(graph::AbstractMatrix{<:Real}, ref_embedding::AbstractMatrix{T<:AbstractFloat}) -> embedding

Initialize an embedding of points corresponding to the columns of the `graph`, by taking weighted means of
the columns of `ref_embedding`, where weights are values from the rows of the `graph`.

The resulting embedding will have shape `(size(ref_embedding, 1), size(graph, 2))`, where `size(ref_embedding, 1)`
is the number of components (dimensions) of the `reference embedding`, and `size(graph, 2)` is the number of 
samples in the resulting embedding. Its elements will have type T.
"""
function initialize_embedding(graph::AbstractMatrix{<:Real}, ref_embedding::AbstractMatrix{T})::Matrix{T} where {T<:AbstractFloat}
    (ref_embedding * graph) ./ (sum(graph, dims=1) .+ eps(T))
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

"""
    optimize_embedding(graph, query_embedding, ref_embedding, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate, _a=nothing, _b=nothing; move_ref=false) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.
Optimize "query" samples with respect to "reference" samples.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `query_embedding`: a matrix (n_components, n_samples) the embedded data points to be optimized ("query" samples)
- `ref_embedding`: a matrix (n_components, n_samples) the embedded data points to be optimized against ("reference" samples)
- `n_epochs`: the number of training epochs for optimization
- `initial_alpha`: the initial learning rate
- `gamma`: the repulsive strength of negative samples
- `neg_sample_rate`: the number of negative samples per positive sample
- `_a`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.
- `_b`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.

# Keyword Arguments
- `move_ref::Bool = false`: if true, also improve the embeddings in `ref_embedding`, else fix them and only improve embeddings in `query_embedding`.
"""
function optimize_embedding(graph::SparseMatrixCSC{T},
                            query_embedding::AbstractMatrix{S},
                            ref_embedding::AbstractMatrix{S},
                            n_epochs::Integer,
                            initial_alpha::Real,
                            min_dist::Real,
                            spread::Real,
                            gamma::Real,
                            neg_sample_rate::Integer,
                            _a::Union{Real, Nothing} = nothing,
                            _b::Union{Real, Nothing} = nothing;
                            move_ref::Bool=false) where {T <: Real, S <: Real}
    a, b = fit_ab(convert(S,min_dist), convert(S, spread), _a, _b)
    self_reference = query_embedding === ref_embedding

    alpha = initial_alpha
    for e in 1:n_epochs
        @inbounds for i in 1:size(graph, 2)
            for ind in nzrange(graph, i)
                j = rowvals(graph)[ind]
                p = nonzeros(graph)[ind]

                if rand() <= p
                    xi = view(query_embedding, :, i)
                    xj = view(ref_embedding, :, j)

                    sdist = evaluate(SqEuclidean(), xi, xj)
                    sdist = max(sdist, eps(S))
                    delta = (-2 * a * b * sdist^b)/(sdist*(1 + a*sdist^b))

                    @simd for d in eachindex(xi)
                        grad = clamp(delta * (xi[d] - xj[d]), -4, 4)
                        xi[d] += alpha * grad
                        if move_ref
                            xj[d] -= alpha * grad
                        end
                    end

                    for _ in 1:neg_sample_rate
                        k = rand(1:size(ref_embedding, 2))
                        if i == k && self_reference
                            continue
                        end
                        xk = view(ref_embedding, :, k)

                        sdist = evaluate(SqEuclidean(), xi, xk)
                        sdist = max(sdist, eps(S))
                        delta = (2 * gamma * b) / ((S(1//1000) + sdist)*(1 + a*sdist^b))

                        @simd for d in eachindex(xi)
                            if delta > 0
                                grad = clamp(delta * (xi[d] - xk[d]), -4, 4)
                            else
                                grad = 4
                            end
                            xi[d] += alpha * grad
                        end
                    end

                end
            end
        end
        alpha = initial_alpha*(1 - e//n_epochs)
    end

    return query_embedding
end
