

"""
    OptimizationParams(n_epochs, learning_rate, repulsion_strength, neg_sample_rate)

Parameters for controlling the optimization process.
"""
struct OptimizationParams
    "The number of epochs to perform optimization"
    n_epochs::Int
    "The initial learning rate for optimization (decreases each epoch)"
    lr::Float64
    "The weighting of negative samples during optimization"
    repulsion_strength::Float64
    """
    The number of negative samples to select for each positive sample.
    Higher values increase computational cost but result in slightly better accuracy.
    """
    neg_sample_rate::Int
    function OptimizationParams(n_epochs, lr, repulsion_strength, neg_sample_rate)
        n_epochs > 0 || throw(ArgumentError("n_epochs must be greater than 0, got $n_epochs"))
        lr ≥ 0 || throw(ArgumentError("learning_rate must be non-negative, got $lr"))
        neg_sample_rate ≥ 0 || throw(ArgumentError("neg_sample_rate must be non-negative, got $neg_sample_rate"))
        return new(n_epochs, lr, repulsion_strength, neg_sample_rate)
    end
end


function optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)
    _opt_params = opt_params
    for e in 1:opt_params.n_epochs
        _optimize_embedding!(embedding, 
                             embedding, 
                             umap_graph, 
                             tgt_params, 
                             _opt_params;
                             move_ref=true)
        alpha = (1 - e / opt_params.n_epochs) * opt_params.lr
        _opt_params = @set _opt_params.lr = alpha
    end

    return embedding
end

function optimize_embedding!(embedding, ref_embedding, umap_graph, tgt_params, opt_params)
    _opt_params = opt_params
    for e in 1:opt_params.n_epochs
        _optimize_embedding!(embedding, 
                             ref_embedding, 
                             umap_graph, 
                             tgt_params, 
                             _opt_params;
                             move_ref=false)
        alpha = (1 - e / opt_params.n_epochs) * opt_params.lr
        _opt_params = @set _opt_params.lr = alpha
    end

    return embedding
end

function _optimize_embedding!(embedding, 
                              ref_embedding, 
                              umap_graph, 
                              tgt_params::TargetParams{_EuclideanManifold{N}, Distances.SqEuclidean, I, MembershipFnParams{T}},
                              opt_params;
                              move_ref::Bool=true) where {N, I, T}
    
    self_reference = embedding === ref_embedding
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b
    lr = opt_params.lr

    for i in 1:size(umap_graph, 2)
        for ind in nzrange(umap_graph, i)
            j = rowvals(umap_graph)[ind]
            p = nonzeros(umap_graph)[ind]
            if rand() <= p
                dist = Distances.sqeuclidean(embedding[i], ref_embedding[j])
                if dist > 0
                    grad_coef = -(a * b) / (dist * (a + dist^(-b)))
                else
                    grad_coef = zero(dist)
                end
                # update embedding according to clipped gradient
                @simd for d in eachindex(embedding[i])
                    grad = clamp(grad_coef * 2 * (embedding[i][d] - ref_embedding[j][d]), -4, 4)
                    embedding[i][d] += lr * grad
                    ref_embedding[j][d] -= move_ref * lr * grad
                end
                # negative samples
                for _ in 1:opt_params.neg_sample_rate
                    k = rand(eachindex(ref_embedding))
                    if i == k && self_reference
                        continue
                    end
                    dist = Distances.sqeuclidean(embedding[i], ref_embedding[k])
                    if dist > 0
                        grad_coef = opt_params.repulsion_strength * b / (a * dist^(b + 1) + dist)
                    else
                        grad_coef = 4 * one(dist)
                    end
                    # update embedding according to clipped gradient
                    @simd for d in eachindex(embedding[i])
                        grad = clamp(grad_coef * 2 * (embedding[i][d] - ref_embedding[k][d]), -4, 4)
                        embedding[i][d] += lr * grad
                    end
                end
            end
        end
    end
    return embedding
end

function _update_embedding_pos!(embedding, 
                                i,
                                j, 
                                move_ref, 
                                tgt_params::TargetParams{_EuclideanManifold{N}, Distances.SqEuclidean},
                                opt_params) where N
    # specialized for sq euclidean metric
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b
    lr = opt_params.lr
    dist = Distances.sqeuclidean(embedding[i], embedding[j])
    if dist > 0
        grad_coef = -(a * b) / (dist * (a + dist^(-b)))
    else
        grad_coef = zero(dist)
    end
    @simd for d in eachindex(embedding[i])
        grad = clamp(grad_coef * 2 * (embedding[i][d] - embedding[j][d]), -4, 4)
        embedding[i][d] += lr * grad
        embedding[j][d] -= move_ref * lr * grad
    end
    return
end

function _update_embedding_pos!(embedding, i, j, move_ref, tgt_params, opt_params)
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b
    dist, dist_lgrad, dist_rgrad = target_metric(tgt_params, embedding[i], embedding[j])
    if dist > 0
        grad_coef = -(a * b) / (dist * (a + dist^(-b)))
    else
        grad_coef = zero(dist)
    end
    # update embedding according to clipped gradient
    embedding[i] += opt_params.lr * clamp.(grad_coef * dist_lgrad, -4, 4)
    if move_ref
        embedding[j] += opt_params.lr * clamp.(grad_coef * dist_rgrad, -4, 4)
    end
    return
end


"""
    target_metric(tgt_params, x, y) -> dist, grad_dist_x, grad_dist_y

Calculate the distance between `x` and `y` on the manifold `tgt_params.manifold` according to 
`tgt_params.metric` as well as the gradient of that distance with respect to x and y.
"""
function target_metric(tgt_params, x, y) end

function target_metric(::TargetParams{_EuclideanManifold{N}, Distances.SqEuclidean}, x, y) where N
    dist = Distances.sqeuclidean(x, y)
    grad_dist = 2 * (x - y)
    return dist, grad_dist, -grad_dist
end

function target_metric(::TargetParams{_EuclideanManifold{N}, Distances.Euclidean}, x, y) where N
    dist = Distances.euclidean(x, y)
    grad_dist = (x - y) / (1e-8 + dist)
    return dist, grad_dist, -grad_dist
end

"""
A smooth approximation for the membership strength of the 1-simplex between two points x, y.
"""
function _ϕ(x, y, σ, a, b)
    return inv(1 + a*σ(x, y)^b)
end

"""
    cross_entropy(umap_graph, embedding, tgt_params)

Calculate the fuzzy cross entropy loss of the embedding for this umap_graph.
(NOTE: μ(x, y) = umap_graph[x, y]; ν(x, y) = )
"""
function cross_entropy(umap_graph, embedding, tgt_params)
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b
    EPS = 1e-8
    loss = zero(eltype(umap_graph))
    # calculate loss for each edge in the graph
    for I in eachindex(umap_graph)
        i, j = Tuple(I)
        mu_a = clamp(umap_graph[I], EPS, 1 - EPS)
        nu_a = clamp(_ϕ(embedding[i], embedding[j], tgt_params.metric, a, b), EPS, 1 - EPS)

        _loss = mu_a * log(mu_a / nu_a) + (1 - mu_a) * log((1 - mu_a) / (1 - nu_a))
        loss += _loss
    end
    return loss
end
