#=
Utilities used by UMAP.jl
=#

# combine local fuzzy simplicial sets
function merge_local_simplicial_sets(fs_set,
                                     set_op_ratio)
    return set_op_ratio .* _fuzzy_set_union(fs_set) .+
           (1 - set_op_ratio) .* _fuzzy_set_intersection(fs_set)
end

function _fuzzy_set_union(fs_set)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

function _fuzzy_set_intersection(fs_set)
    return fs_set .* fs_set'
end

"""
    general_simplicial_set_union(left_view, right_view, params)

Take the union of two global fuzzy simplicial sets.
"""
function general_simplicial_set_union(left_view, right_view)
    
    result = left_view + right_view

    left_min = max(minimum(left_view) / 2, 1e-8)
    right_min = max(minimum(right_view) / 2, 1e-8)

    for ind in findall(!iszero, result)
        # for each index that is nonzero in at least one of left/right
        left_val = max(left_min, left_view[ind])
        right_val = max(right_min, right_view[ind])
        result[ind] = left_val + right_val - left_val * right_val
    end
    return result

end

"""
    general_simplicial_set_intersection(left_view, right_view, global_params)

Take the weighted intersection of two fuzzy simplicial sets (represented as (sparse) matrices).
Since we don't want to completely lose edges that are only present in one of the two sets, we 
multiply by at least `1e-8`. Furthermore, if the same edge in both sets has a strength below
1e-8, these are added together instead of multiplying.
"""
function general_simplicial_set_intersection(left_view, right_view, params::SourceGlobalParams)
    # start with adding - this gets us a sparse matrix whose nonzero entries
    # are the union of left and right entries
    result = left_view + right_view
    # these added values are replaced by the weighted intersection except when both left_view[ind] and
    # right_view[ind] are less than left_min/right_min respectively.
    left_min = max(minimum(left_view) / 2, 1e-8)
    right_min = max(minimum(right_view) / 2, 1e-8)
    #
    for ind in findall(!iszero, result)
        # take the weighted intersection of the two sets, making sure not to
        # zero out any results by setting minimum values
        left_val = max(left_min, left_view[ind])
        right_val = max(right_min, right_view[ind])
        if left_val > left_min || right_val > right_min
            result[ind] = _mix_values(left_val, right_val, params.mix_ratio)
        end
    end
    return result
end

function _mix_values(x, y, ratio)
    if ratio < 0.5
        return x * y^(ratio / (1 - ratio))
    else
        return x^((1 - ratio) / ratio) * y
    end
end


"""
    reset_local_connectivity(simplicial_set, reset_local_metrics)

Reset the local connectivity requirement -- each data sample should
have complete confidence in at least one 1-simplex in the simplicial set.
We can enforce this by locally rescaling confidences, and then remerging the
different local simplicial sets together.
"""
function reset_local_connectivity(simplicial_set, reset_local_metrics)
    # normalize columns 
    sset = _norm_sparse(simplicial_set)
    
    if reset_local_metrics

    end

    sset = _fuzzy_set_union(sset)

    return dropzeros(sset)

end

function _norm_sparse(simplicial_set::AbstractSparseMatrix)
    # normalize columns of sparse matrix so max is 1 for each column
    maxvals = maximum(simplicial_set, dims=1)
    I, J, V = findnz(simplicial_set)
    newVs = copy(V)

    for (ind, col) in enumerate(J)
        newVs[ind] /= max(maxvals[col], 1e-8) # to avoid division by zero
    end
    return sparse(I, J, newVs, simplicial_set.m, simplicial_set.n)
end

function reset_local_metrics(simplicial_set)
    
end

function _reprocess_col() end