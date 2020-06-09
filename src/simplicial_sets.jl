# creating fuzzy simplicial set representations of data

"""
    merge_view_sets(view_fuzzy_sets, params)

Merge the fuzzy simplicial sets for each view of the data. This returns a single
fuzzy simplicial set - the weighted, undirected UMAP graph - that captures the
spatial relationships of the data points approximated by the manifold.
"""
function merge_view_sets end

function merge_view_sets(view_fuzzy_sets::NamedTuple{T},
                         gbl_params::SourceGlobalParams) where T
    # TODO
end

# if no global params are passed, there must be exactly one view in the named
# tuple - dispatch here.
function merge_view_sets(view_fuzzy_sets::NamedTuple{R, T},
                         ::Nothing) where {R, S, T <: Tuple{S}}
    return view_fuzzy_sets[T[1]]
end

"""
    fuzzy_simplicial_set
"""
# fit
function fuzzy_simplicial_set(view_knns::NamedTuple{T},
                              src_params::NamedTuple{T}) where T
    view_fuzzy_sets = map(fuzzy_simplicial_set, view_knns, src_params)
    return view_fuzzy_sets
end

# TODO: transform
function fuzzy_simplicial_set(result,
                              view_knns::NamedTuple{T},
                              src_params::NamedTuple{T}) where T
    view_fuzzy_sets = map(fuzzy_simplicial_set, result, view_knns, src_params)
    return view_fuzzy_sets
end

# create global fuzzy simplicial set for a single view (fit)
function fuzzy_simplicial_set((knns, dists),
                              src_params::SourceViewParams)

    σs, ρs = smooth_knn_dists(dists, size(knns, 1), src_params)

    rows, cols, vals = compute_membership_strengths(knns, dists, σs, ρs)
    local_fs_sets = sparse(rows, cols, vals, size(knns, 2), size(knns, 2))

    fs_set = combine_fuzzy_sets(local_fs_sets, src_params.set_operation_ratio)
    return dropzeros(fs_set)
end

# SIMPLICIAL SET UTILITIES

const SMOOTH_K_TOLERANCE = 1e-5

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
    for i in 1:size(knn_dists, 2)
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
@fastmath function smooth_knn_dist(dists::AbstractVector, ρ, k, bandwidth, niter=64)
    target = log2(k)*bandwidth
    lo, mid, hi = 0., 1., Inf
    for n in 1:niter
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
    for i in 1:size(knns, 2), j in 1:size(knns, 1)
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
