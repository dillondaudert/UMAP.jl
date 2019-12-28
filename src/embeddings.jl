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
    optimize_embedding(graph, embedding, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `embedding`: a vector of length (n_samples,) of vectors representing the embedded data points
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
                    sdist = evaluate(SqEuclidean(), embedding[i], embedding[j])
                    if sdist > 0
                        delta = (-2 * a * b * sdist^(b-1))/(1 + a*sdist^b)
                    else
                        delta = 0
                    end
                    @simd for d in eachindex(embedding[i])
                        grad = clamp(delta * (embedding[i][d] - embedding[j][d]), -4, 4)
                        embedding[i][d] += alpha * grad
                        embedding[j][d] -= alpha * grad
                    end

                    for _ in 1:neg_sample_rate
                        k = rand(1:size(graph, 2))
                        i != k || continue
                        sdist = evaluate(SqEuclidean(), embedding[i], embedding[k])
                        if sdist > 0
                            delta = (2 * gamma * b) / ((1//1000 + sdist)*(1 + a*sdist^b))
                        else
                            delta = 0
                        end
                        @simd for d in eachindex(embedding[i])
                            if delta > 0
                                grad = clamp(delta * (embedding[i][d] - embedding[k][d]), -4, 4)
                            else
                                grad = 4
                            end
                            embedding[i][d] += alpha * grad
                        end
                    end

                end
            end
        end
        alpha = initial_alpha*(1 - e//n_epochs)
    end

    return embedding
end
