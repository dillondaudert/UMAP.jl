#=
Utilities used by UMAP.jl
=#

# combine local fuzzy simplicial sets
function combine_fuzzy_sets(fs_set,
                            set_op_ratio)
    return set_op_ratio .* fuzzy_set_union(fs_set) .+
           (1 - set_op_ratio) .* fuzzy_set_intersection(fs_set)
end

function fuzzy_set_union(fs_set)
    return fs_set .+ fs_set' .- (fs_set .* fs_set')
end

function fuzzy_set_intersection(fs_set)
    return fs_set .* fs_set'
end

function fuzzy_set_intersection(left_view, right_view, params)
    # start with adding - this gets us a sparse matrix whose nonzero entries
    # are the union of left and right entries
    result = left_view .+ right_view
    left_min = max(minimum(left_view.nzval) / 2, 1e-8)
    right_min = max(minimum(right_view.nzval) / 2, 1e-8)
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
