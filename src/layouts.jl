
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
function optimize_embedding!(graph,
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
        embedding = _optimize_embedding!(graph, embedding, alpha, min_dist, spread, gamma, neg_sample_rate, a, b)
        alpha = initial_alpha*(1 - e//n_epochs)
    end

    return embedding
end

@views function _optimize_embedding!(graph, embedding, alpha, min_dist, spread, gamma, neg_sample_rate, a, b)
    @inbounds for i in 1:size(graph, 2)
        for ind in nzrange(graph, i)
            j = rowvals(graph)[ind]
            p = nonzeros(graph)[ind]
            if rand() <= p
                delta = pos_grad_coef(SqEuclidean(), embedding[:, i], embedding[:, j], a, b)
                @simd for d in 1:size(embedding, 1)
                    grad = clamp(delta * (embedding[d,i] - embedding[d,j]), -4, 4)
                    embedding[d,i] += alpha * grad
                    embedding[d,j] -= alpha * grad
                end

                for _ in 1:neg_sample_rate
                    k = rand(1:size(graph, 2))
                    i != k || continue # don't evaluate if the same point
                    delta = neg_grad_coef(SqEuclidean(), embedding[:, i], embedding[:, k])
                    @simd for d in 1:size(embedding, 1)
                        if delta > 0
                            grad = clamp(delta * (embedding[d, i] - embedding[d, k]), -4, 4)
                            embedding[d, i] += alpha * grad
                        else
                            embedding[d, i] += alpha * 4
                        end
                    end
                end

            end
        end
    end
    return embedding
end

function pos_grad_coef(metric, head::V, tail::V, a, b) where {V <: AbstractVector}
    dist = evaluate(metric, head, tail)
    delta = (-2 * a * b * dist^(b-1))/(1 + a*dist^b)
    return max(zero(eltype(head)), delta)
end

function neg_grad_coef(metric, head::V, tail::V, gamma, a, b) where {V <: AbstractVector}
    dist = evaluate(metric, head, tail)
    delta = (2 * gamma * b) / ((dist + 1//1000)*(1 + a * dist^b))
    return max(zero(eltype(head)), delta)
end
