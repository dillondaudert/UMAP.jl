# creating fuzzy simplicial set representations of data

# CONSTANTS
"""
    SMOOTH_K_TOLERANCE

Tolerance for the smooth k-distance calculation.
"""
const SMOOTH_K_TOLERANCE = 1.0e-5

# SOURCE PARAMS
"""
    SourceViewParams(set_operation_ratio, local_connectivity, bandwidth)

Struct for parameterizing the representation of the data in the source (original)
manifold; i.e. constructing fuzzy simplicial sets of each view of the dataset.
"""
struct SourceViewParams{T <: Real}
    """
    The ratio of set union to set intersection used to combine local fuzzy simplicial sets, 
    from 0 (100% intersection) to 1 (100% union)
    """
    set_operation_ratio::T
    """
    The number of nearest neighbors that should be assumed to be locally connected. 
    The higher this value, the more connected the manifold becomes. 
    This should not be set higher than the intrinsic dimension of the manifold.
    """
    local_connectivity::T
    "bandwidth"
    bandwidth::T
    function SourceViewParams{T}(set_op_ratio, local_conn, bandwidth) where {T <: Real}
        0 ≤ set_op_ratio ≤ 1 || throw(ArgumentError("set_op_ratio must be between 0 and 1"))
        local_conn > 0 || throw(ArgumentError("local_connectivity must be greater than 0"))
        bandwidth > 0 || throw(ArgumentError("bandwidth must be greater than 0"))
        return new(set_op_ratio, local_conn, bandwidth)
    end
end
function SourceViewParams(set_op_ratio::T, local_conn::T, bandwidth::T) where {T <: Real}
    return SourceViewParams{T}(set_op_ratio, local_conn, bandwidth)
end
function SourceViewParams(set_op_ratio::Real, local_conn::Real, bandwidth::Real)
    return SourceViewParams(promote(set_op_ratio, local_conn, bandwidth)...)
end

"""
    SourceGlobalParams{T}(mix_ratio)

Parameters for merging the fuzzy simplicial sets for each dataset view into one
fuzzy simplicial set, otherwise known as the UMAP graph.
"""
struct SourceGlobalParams{T <: Real}
    mix_ratio::T
    function SourceGlobalParams{T}(mix_ratio) where {T <: Real}
        0 ≤ mix_ratio ≤ 1 || throw(ArgumentError("mix_ratio must be between 0 and 1"))
        return new(mix_ratio)
    end
end
SourceGlobalParams(mix_ratio::T) where {T <: Real} = SourceGlobalParams{T}(mix_ratio)


"""
    coalesce_views(view_fuzzy_sets, params)

Merge the fuzzy simplicial sets for each view of the data. This returns a single
fuzzy simplicial set - the weighted, undirected UMAP graph - that captures the
spatial relationships of the data points approximated by the manifold.
"""
function coalesce_views end

function coalesce_views(view_fuzzy_sets::NamedTuple{T},
                        gbl_params) where T
    return foldl((l, r) -> reset_local_connectivity(general_simplicial_set_intersection(l, r, gbl_params)), view_fuzzy_sets)
end

# if no global params are passed, there must be exactly one view in the named
# tuple - dispatch here.
function coalesce_views(view_fuzzy_sets,
                         _)
    return view_fuzzy_sets
end

"""
Construct the UMAP graph, i.e. the global fuzzy simplicial set. This is
represented as a symmetric, sparse matrix where each value is the 
probability that an edge exists between points (row, col).

For multiple views, there will be one UMAP graph per view - later combined
via `coalesce_views`. 

For transforming new data, this graph is notably not symmetric.
"""
function fuzzy_simplicial_set end

"""
    fuzzy_simplicial_set(knns_dists::NamedTuple{T}, knn_params::NamedTuple{T}, src_params::NamedTuple{T}) -> NamedTuple{T}

(Fit) Construct a fuzzy simplicial set for each view of the data, returning a named tuple
of fuzzy simplicial sets, one for each view. The keys of the named tuple are the
keys of the input NamedTuples.
"""
function fuzzy_simplicial_set(knns_dists::NamedTuple{T},
                              knn_params::NamedTuple{T},
                              src_params::NamedTuple{T}) where T
    return map(fuzzy_simplicial_set, knns_dists, knn_params, src_params)
end

"""
    fuzzy_simplicial_set(data::NamedTuple{T}, knns_dists::NamedTuple{T}, knn_params::NamedTuple{T}, src_params::NamedTuple{T}) -> NamedTuple{T}

(Transform) Construct a fuzzy simplicial set for each view of the data, returning a named tuple
of fuzzy simplicial sets, one for each view.
"""
function fuzzy_simplicial_set(data::NamedTuple{T},
                              knns_dists::NamedTuple{T},
                              knn_params::NamedTuple{T},
                              src_params::NamedTuple{T}) where T
    return map(fuzzy_simplicial_set, data, knns_dists, knn_params, src_params)
end

"""
    fuzzy_simplicial_set((knns, dists), knn_params, src_params::SourceViewParams)

(Fit) Construct a global fuzzy simplicial set for a single data view.
"""
function fuzzy_simplicial_set((knns, dists), knn_params, src_params::SourceViewParams)
    n_points = size(knns, 2)
    return fuzzy_simplicial_set((knns, dists), n_points, knn_params, src_params, true)
end

function fuzzy_simplicial_set(data, knns_dists, knn_params, src_params::SourceViewParams)
    # the length of the last axis tells us how many points there are
    n_points = length(axes(data)[end])
    return fuzzy_simplicial_set(knns_dists, n_points, knn_params, src_params, false)
end

"""
    fuzzy_simplicial_set(knns_dists, n_points, knn_params, src_params, combine=true) -> membership_graph::SparseMatrixCSC

Construct the local fuzzy simplicial set of each point represented by its distances
to its nearest neighbors, stored in `knns` and `dists`, normalizing the distances,
and converting the metric space to a simplicial set (a weighted graph).

`n_points` indicates the total number of points of the original data, while `knns` contains
indices of some subset of those points (ie some subset of 1:`n_points`). If `knns` represents
neighbors of the elements of some set with itself, then `knns` should have `n_points` number of
columns. Otherwise, these two values may be different.

If `combine` is true, use intersections and unions to combine local fuzzy sets of neighbors.
The returned graph has size (`n_points`, size(knns, 2)).
"""
function fuzzy_simplicial_set((knns, dists),
                              n_points::Integer,
                              knn_params::NeighborParams,
                              src_params::SourceViewParams,
                              combine::Bool)

    σs, ρs = smooth_knn_dists(dists, knn_params.n_neighbors, src_params)

    rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    local_fs_sets = sparse(rows, cols, vals, n_points, size(knns, 2))

    if combine
        fs_set = merge_local_simplicial_sets(local_fs_sets, src_params.set_operation_ratio)
    else
        fs_set = local_fs_sets
    end
    return dropzeros(fs_set)
end

# SIMPLICIAL SET UTILITIES

"""
    smooth_knn_dists(dists, k, src_params) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.
"""
function smooth_knn_dists(knn_dists::AbstractMatrix{S},
                          k,
                          src_params::SourceViewParams) where {S <: Real}
    local_connectivity = src_params.local_connectivity
    nonzero_dists(dists) = @view dists[dists .> 0.]
    ρs = zeros(S, size(knn_dists, 2))
    σs = Array{S}(undef, size(knn_dists, 2))
    for i in axes(knn_dists, 2)
        nz_dists = nonzero_dists(knn_dists[:, i])
        if length(nz_dists) >= local_connectivity
            index = floor(Int, local_connectivity)
            interpolation = local_connectivity - index
            if index > 0
                ρs[i] = nz_dists[index]
                if interpolation > SMOOTH_K_TOLERANCE
                    ρs[i] += interpolation * (nz_dists[index+1] - nz_dists[index])
                end
            else
                ρs[i] = interpolation * nz_dists[1]
            end
        elseif length(nz_dists) > 0
            ρs[i] = maximum(nz_dists)
        end
        @inbounds σs[i] = smooth_knn_dist(knn_dists[:, i], ρs[i], k, src_params.bandwidth)
    end

    return ρs, σs
end

# calculate sigma for an individual point
function smooth_knn_dist(dists::AbstractVector, ρ, k, bandwidth, niter=64)
    target = log2(k)*bandwidth
    lo, mid, hi = 0., 1., Inf
    for _ in 1:niter
        psum = sum(exp.(-max.(dists .- ρ, 0.)./mid))
        if abs(psum - target) < SMOOTH_K_TOLERANCE
            break
        end
        if psum > target
            hi = mid
            mid = (lo + hi)/2.
        else
            lo = mid
            if hi == Inf
                mid *= 2.
            else
                mid = (lo + hi) / 2.
            end
        end
    end
    # TODO: set according to min k dist scale
    return mid
end

"""
    compute_membership_strengths(knns, dists, σs, ρs) -> rows, cols, vals

Compute the membership strengths for the 1-skeleton of each fuzzy simplicial set.
"""
function compute_membership_strengths(knns::AbstractMatrix{S},
                                      dists::AbstractMatrix{T},
                                      ρs::Vector{T},
                                      σs::Vector{T}) where {S <: Integer, T}
    # set dists[i, j]
    rows = sizehint!(S[], length(knns))
    cols = sizehint!(S[], length(knns))
    vals = sizehint!(T[], length(knns))
    for i in axes(knns, 2), j in axes(knns, 1)
        @inbounds if i == knns[j, i] # dist to self
            d = 0.
        else
            @inbounds d = exp(-max(dists[j, i] - ρs[i], 0.)/σs[i])
        end
        append!(cols, i)
        append!(rows, knns[j, i])
        append!(vals, d)
    end
    return rows, cols, vals
end


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
    general_simplicial_set_union(left_view, right_view)

Take the union of two _global_ fuzzy simplicial sets.
"""
function general_simplicial_set_union(left_view::M, right_view::M) where {M <: AbstractSparseMatrix}
    
    result = left_view + right_view

    left_min = max(minimum(nonzeros(left_view)) / 2, 1e-8)
    right_min = max(minimum(nonzeros(right_view)) / 2, 1e-8)

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
function general_simplicial_set_intersection(left_view::M, right_view::M, params::SourceGlobalParams) where {M <: AbstractSparseMatrix}
    # start with adding - this gets us a sparse matrix whose nonzero entries
    # are the union of left and right entries
    result = left_view + right_view
    # these added values are replaced by the weighted intersection except when both left_view[ind] and
    # right_view[ind] are less than left_min/right_min respectively.
    left_min = max(minimum(nonzeros(left_view)) / 2, 1e-8)
    right_min = max(minimum(nonzeros(right_view)) / 2, 1e-8)
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

"""
    _mix_values(x, y, ratio)

For ratio in [0, 1], return a modified multiplication of x and y.
For ratio = 0, return x
For ratio = 1, return y
For ratio = 0.5, return x*y
"""
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

A FS set may lose this property when combining multiple views of the source
data; this function restores it.
"""
function reset_local_connectivity(simplicial_set, reset_local_metric=true)
    # normalize columns 
    sset = _norm_sparse(simplicial_set)
    
    if reset_local_metric
        sset = reset_local_metrics!(sset)
    end

    sset = _fuzzy_set_union(sset)

    return dropzeros(sset)

end

"""
    _norm_sparse(fsset)

For each column (i.e. each point), normalize the membership values 
(divide by the maximum). This creates a copy of the matrix.
"""
function _norm_sparse(simplicial_set::SparseMatrixCSC)
    # normalize columns of sparse matrix so max is 1 for each column
    maxvals = maximum(simplicial_set, dims=1)
    I, J, V = findnz(simplicial_set)
    newVs = copy(V)

    for (ind, col) in enumerate(J)
        newVs[ind] /= max(maxvals[col], 1e-8) # to avoid division by zero
    end
    return sparse(I, J, newVs, simplicial_set.m, simplicial_set.n)
end

function reset_local_metrics!(simplicial_set::SparseMatrixCSC)

    # for each column, reset the fuzzy set cardinality.
    # modify each sparse column in-place here.
    for col in axes(simplicial_set)[end]
        colstart, colend = simplicial_set.colptr[col], simplicial_set.colptr[col+1]-1
        simplicial_set.nzval[colstart:colend] .= _reset_fuzzy_set_cardinality(simplicial_set.nzval[colstart:colend])
    end

    return simplicial_set
end

"""
Reset the cardinality of the fuzzy set (usually a column
in a simplicial set) to be approximately log2(k).

This step is necessary after we've combined the simplicial sets
for multiple views of the same data.
"""
function _reset_fuzzy_set_cardinality(probs, k=15, niter=32)
    target = log2(k)

    lo = 0.
    hi = Inf
    mid = 1.

    for _ in 1:niter
        psum = sum(probs .^ mid)
        if abs(psum - target) < SMOOTH_K_TOLERANCE
            break
        end

        if psum < target
            hi = mid
            mid = (lo + hi) / 2
        else
            lo = mid
            if isinf(hi)
                mid *= 2
            else
                mid = (lo + hi) / 2
            end
        end
    end
    return probs .^ mid
end