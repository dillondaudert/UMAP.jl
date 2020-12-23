
function optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)
    _opt_params = opt_params
    for e in opt_params.n_epochs
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
    for e in opt_params.n_epochs
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
                              tgt_params::TargetParams,
                              opt_params::OptimizationParams;
                              move_ref::Bool=true)
    
    self_reference = embedding === ref_embedding
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b

    for i in 1:size(umap_graph, 2)
        for ind in nzrange(umap_graph, i)
            j = rowvals(umap_graph)[ind]
            p = nonzeros(umap_graph)[ind]
            if rand() <= p
                dist, dist_lgrad, dist_rgrad = target_metric(tgt_params, embedding[i], ref_embedding[j])
                if dist > 0
                    grad_coef = -(a * b) / (dist * (a + dist^(-b)))
                else
                    grad_coef = 0
                end
                # update embedding according to clipped gradient
                embedding[i] .+= opt_params.lr .* clamp.(grad_coef .* dist_lgrad, -4, 4)
                if move_ref
                    ref_embedding[j] .+= opt_params.lr .* clamp.(grad_coef .* dist_rgrad, -4, 4)
                end
                # negative samples
                for _ in 1:opt_params.neg_sample_rate
                    k = rand(eachindex(ref_embedding))
                    if i == k && self_reference
                        continue
                    end
                    dist, dist_lgrad, _ = target_metric(tgt_params, embedding[i], ref_embedding[k])
                    if dist > 0
                        grad_coef = opt_params.repulsion_strength * b / (a * dist^(b + 1) + dist)
                    else
                        grad_coef = 0
                    end
                    # update embedding according to clipped gradient
                    embedding[i] .+= opt_params.lr .* clamp.(grad_coef .* dist_lgrad, -4, 4)
                end
            end
        end
    end
    return embedding
end

"""
    target_metric(tgt_params, x, y) -> dist, grad_dist_x, grad_dist_y

Calculate the distance between `x` and `y` on the manifold `tgt_params.manifold` according to 
`tgt_params.metric` as well as the gradient of that distance with respect to x and y.
"""
function target_metric(tgt_params, x, y) end

function target_metric(::TargetParams{_EuclideanManifold{N}, SqEuclidean}, x, y) where N
    dist = SqEuclidean()(x, y)
    grad_dist = 2 * (x - y)
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
