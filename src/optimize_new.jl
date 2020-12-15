
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
    
    self_ref = embedding === ref_embedding
    a, b = tgt_params.memb_params.a, tgt_params.memb_params.b

    for i in 1:size(umap_graph, 2)
        for ind in nzrange(umap_graph, i)
            j = rowvals(umap_graph)[ind]
            p = nonzeros(umap_graph)[ind]
            if rand() <= p
                dist, dist_lgrad, dist_rgrad = target_metric(tgt_params, embedding[i], ref_embedding[j])
                if dist > 0
                    w_l = inv(1 + a * dist^(2*b))
                else
                    w_l = 1
                end
                grad_coef = 2 * b * (w_l - 1) / (dist + 1e-6)
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
                        w_l = inv(1 + a * dist^(2*b))
                    else
                        w_l = 1
                    end
                    grad_coef = opt_params.repulsion_strength * 2 * b * w_l / (dist + 1e-6)
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