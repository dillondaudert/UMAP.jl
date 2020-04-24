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
    initialize_embedding(graph, ref_embedding, query_inds::Vector{<:Integer}, ref_inds::Vector{<:Integer}) -> embedding

Initialize an embedding of points corresponding to the `query_inds` columns of the `graph`, by taking weighted means of
the `ref_embedding`, where weights are values from the `ref_inds` rows of the `graph`.

`ref_inds[i]` is the row in the `graph` corresponding to the `i`th sample of `ref_embedding`, 
ie. the sample `ref_embedding[:, i]`.

The resulting embedding will have shape `(size(ref_embedding, 1), length(query_inds))`, where `size(ref_embedding, 1)`
is the number of components (dimensions) of the `reference embedding`, and `length(query_inds)` is the number of 
samples in the resulting embedding.
"""
function initialize_embedding(graph, ref_embedding, query_inds::Vector{<:Integer}, ref_inds::Vector{<:Integer})
    embed = [zeros(size(ref_embedding, 1)) for _ in 1:length(query_inds)]
    rq_graph = graph[ref_inds, query_inds]
    for (i, col) in enumerate(eachcol(rq_graph))
        embed[i] = vec(sum(transpose(col) .* ref_embedding, dims=2) ./ sum(col))
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
    # query_inds and ref_inds are each the entire set of indices of the embedding, so all samples of the embedding will
    #   be optimized, with respect to each (neighbor) sample.
    query_inds = ref_inds = collect(eachindex(embedding))
    return optimize_embedding(
        graph,
        embedding,
        query_inds,
        ref_inds,
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
    optimize_embedding_rq(graph, embedding,query_inds, ref_inds, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate, _a=nothing, _b=nothing, move_ref=false) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.
Optimize "query" samples with respect to "reference" samples.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `embedding`: a vector of length (n_samples,) of vectors representing the embedded data points
- `query_inds`: the indices of the embedding to optimize.
- `ref_inds`: the indices of the embedding to optimize with respect to.
- `n_epochs`: the number of training epochs for optimization
- `initial_alpha`: the initial learning rate
- `gamma`: the repulsive strength of negative samples
- `neg_sample_rate::Integer`: the number of negative samples per positive sample
- `move_ref::Bool = false`: if true, also improve the embeddings of samples in `ref_inds`, else fix them and only improve embeddings of samples in `query_inds`.
"""
function optimize_embedding(graph,
                            embedding,
                            query_inds,
                            ref_inds,
                            n_epochs,
                            initial_alpha,
                            min_dist,
                            spread,
                            gamma,
                            neg_sample_rate,
                            _a,
                            _b;
                            move_ref::Bool=false)
    a, b = fit_ab(min_dist, spread, _a, _b)
    ref_indset = Set(ref_inds)

    alpha = initial_alpha
    for e in 1:n_epochs
        @inbounds for i in query_inds
            for ind in nzrange(graph, i)
                j = rowvals(graph)[ind]
                if !(j in ref_indset)
                    continue
                end
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
                        if move_ref
                            embedding[j][d] -= alpha * grad
                        end
                    end

                    for _ in 1:neg_sample_rate
                        k = rand(ref_inds)
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
