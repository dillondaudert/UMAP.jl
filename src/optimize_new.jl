
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
    for i in 1:size(umap_graph, 2)
        for ind in nzrange(umap_graph, i)
            j = rowvals(umap_graph)[ind]
            p = nonzeros(umap_graph)[ind]
            if rand() <= p
                lgrad, rgrad = pos_grad(tgt_params, embedding[i], ref_embedding[j])
                update_embedding!(embedding, i, lgrad, tgt_params, opt_params)
                if move_ref
                    update_embedding!(ref_embedding, j, rgrad, tgt_params, opt_params)
                end
                
                for _ in 1:opt_params.neg_sample_rate
                    k = rand(eachindex(ref_embedding))
                    if i == k && self_reference
                        continue
                    end
                    lgrad, _ = neg_grad(tgt_params, embedding[i], ref_embedding[k])
                    update_embedding!(embedding, i, lgrad, tgt_params, opt_params)
                end
            end
        end
    end
    return embedding
end