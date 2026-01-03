

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
        repulsion_strength >= 0 || throw(ArgumentError("repulsion_strength must be non-negative, got $repulsion_strength"))
        return new(n_epochs, lr, repulsion_strength, neg_sample_rate)
    end
end

function set_lr(params::OptimizationParams, lr)
    return Accessors.@set params.lr = lr
end

# FIT
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
        _opt_params = set_lr(_opt_params, alpha)
    end

    return embedding
end

# TRANSFORM
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
        _opt_params = set_lr(_opt_params, alpha)
    end

    return embedding
end

"""
    _optimize_embedding!(embedding, ref_embedding, umap_graph, tgt_params, opt_params; move_ref)

Optimize the embedding for one epoch, calculating the distances between neighbors and 
updating the embedding via gradient descent. If `embedding` and `ref_embedding` are the 
same object, then this is fitting a new embedding. Otherwise, `ref_embedding` is the 
result of a previous call to fit, and we are transforming new data. 

In both cases, the dimensions of `umap_graph` have to match `embedding` and 
`ref_embedding`: umap_graph in R^{n, m}, embedding length m, ref_embedding length n.

This optimizes the default case, where the embeddings are in vector of vectors format,
the target manifold is a Euclidean manifold of some dimension N, and the target metric 
is squared euclidean.
"""
function _optimize_embedding!(embedding::AbstractVector{E}, 
                              ref_embedding::AbstractVector{E}, 
                              umap_graph::SparseMatrixCSC, 
                              tgt_params::TargetParams{_EuclideanManifold{N}, Distances.SqEuclidean},
                              opt_params::OptimizationParams;
                              move_ref::Bool=true) where {E<:AbstractVector, N}

    self_reference = embedding === ref_embedding

    for i in 1:size(umap_graph, 2)
        for ind in nzrange(umap_graph, i)
            j = rowvals(umap_graph)[ind]
            p = nonzeros(umap_graph)[ind]
            if rand() <= p
                update_embedding_pos!(embedding[i], ref_embedding[j], tgt_params, opt_params, move_ref)
                # negative samples
                for _ in 1:opt_params.neg_sample_rate
                    k = rand(eachindex(ref_embedding))
                    if i == k && self_reference
                        # don't calculate negative force with itself
                        continue
                    end
                    update_embedding_neg!(embedding[i], ref_embedding[k], tgt_params, opt_params)
                end
            end
        end
    end
    return embedding
end

"""
Calculate the gradients of the positive 1-simplices in the simplicial set,
and update the embeddings. This assumes embedded in R^d with the 
squared euclidean metric.
"""
function update_embedding_pos!(emb_v::AbstractVector{T}, 
                               emb_w::AbstractVector{T}, 
                               tgt_params::TargetParams{_EuclideanManifold{N}, Distances.SqEuclidean},
                               opt_params::OptimizationParams,
                               move_ref::Bool) where {T <: Real, N}
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b
    lr = opt_params.lr
    dist = Distances.sqeuclidean(emb_v, emb_w)
    if dist > 0
        grad_coef = -(a * b) / (dist * (a + dist^(-b)))
    else
        grad_coef = zero(dist)
    end
    # update embedding according to clipped gradient
    @simd for d in eachindex(emb_v)
        grad = clamp(grad_coef * 2 * (emb_v[d] - emb_w[d]), -4, 4)
        emb_v[d] += lr * grad
        emb_w[d] -= move_ref * lr * grad
    end
    return
end

"""
Calculate the gradients of the negative 1-simplices in the simplicial set,
and update the embeddings. This assumes embedded in R^d with the 
squared euclidean metric.
"""
function update_embedding_neg!(emb_v::AbstractVector{T}, 
                               emb_w::AbstractVector{T}, 
                               tgt_params::TargetParams{_EuclideanManifold{N}, Distances.SqEuclidean},
                               opt_params::OptimizationParams) where {T <: Real, N}
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b
    lr = opt_params.lr
    dist = Distances.sqeuclidean(emb_v, emb_w)
    if dist > 0
        grad_coef = opt_params.repulsion_strength * b / (a * dist^(b + 1) + dist)
    else
        grad_coef = 4 * one(dist)
    end
    # update embedding according to clipped gradient
    @simd for d in eachindex(emb_v)
        grad = clamp(grad_coef * 2 * (emb_v[d] - emb_w[d]), -4, 4)
        emb_v[d] += lr * grad
    end
    return
end


#
# Below, incomplete generic gradient update code for other target 
# metrics, and eventually manifolds.
#

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
function target_metric end

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

#
# Utils
#

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
