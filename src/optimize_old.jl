


"""
    optimize_embedding(graph, query_embedding, ref_embedding, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate, _a=nothing, _b=nothing; move_ref=false) -> embedding

Optimize an embedding by minimizing the fuzzy set cross entropy between the high and low dimensional simplicial sets using stochastic gradient descent.
Optimize "query" samples with respect to "reference" samples.

# Arguments
- `graph`: a sparse matrix of shape (n_samples, n_samples)
- `query_embedding`: a vector of length (n_samples,) of vectors representing the embedded data points to be optimized ("query" samples)
- `ref_embedding`: a vector of length (n_samples,) of vectors representing the embedded data points to optimize against ("reference" samples)
- `n_epochs`: the number of training epochs for optimization
- `initial_alpha`: the initial learning rate
- `gamma`: the repulsive strength of negative samples
- `neg_sample_rate`: the number of negative samples per positive sample
- `_a`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.
- `_b`: this controls the embedding. If the actual argument is `nothing`, this is determined automatically by `min_dist` and `spread`.

# Keyword Arguments
- `move_ref::Bool = false`: if true, also improve the embeddings in `ref_embedding`, else fix them and only improve embeddings in `query_embedding`.
"""
function optimize_embedding(graph,
                            query_embedding,
                            ref_embedding,
                            n_epochs,
                            initial_alpha,
                            min_dist,
                            spread,
                            gamma,
                            neg_sample_rate,
                            _a=nothing,
                            _b=nothing;
                            move_ref::Bool=false)
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
