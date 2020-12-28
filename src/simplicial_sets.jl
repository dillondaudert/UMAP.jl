# creating fuzzy simplicial set representations of data

"""
    coalesce_views(view_fuzzy_sets, params)

Merge the fuzzy simplicial sets for each view of the data. This returns a single
fuzzy simplicial set - the weighted, undirected UMAP graph - that captures the
spatial relationships of the data points approximated by the manifold.
"""
function coalesce_views end

function coalesce_views(view_fuzzy_sets::NamedTuple{T},
                        gbl_params) where T
    return foldl((l, r) -> fuzzy_set_intersection(l, r, gbl_params), view_fuzzy_sets)
end

# if no global params are passed, there must be exactly one view in the named
# tuple - dispatch here.
function coalesce_views(view_fuzzy_sets,
                         _)
    return view_fuzzy_sets
end

"""
    fuzzy_simplicial_set(knns_dists, knn_params, src_params)
    fuzzy_simplicial_set(data, knns_dists, knn_params, src_params)
"""
function fuzzy_simplicial_set end

"""
    fuzzy_simplicial_set(knns_dists::NamedTuple{T}, knn_params::NamedTuple{T}, src_params::NamedTuple{T}) -> NamedTuple{T}
"""
function fuzzy_simplicial_set(knns_dists::NamedTuple{T},
                              knn_params::NamedTuple{T},
                              src_params::NamedTuple{T}) where T
    view_fuzzy_sets = map(fuzzy_simplicial_set, knns_dists, knn_params, src_params)
    return view_fuzzy_sets
end

# transform multiple views
function fuzzy_simplicial_set(data::NamedTuple{T},
                              knns_dists::NamedTuple{T},
                              knn_params::NamedTuple{T},
                              src_params::NamedTuple{T}) where T
    view_fuzzy_sets = map(fuzzy_simplicial_set, data, knns_dists, knn_params, src_params)
    return view_fuzzy_sets
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
columns. Otherwise, these two values may be inequivalent.

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
        fs_set = combine_fuzzy_sets(local_fs_sets, src_params.set_operation_ratio)
    else
        fs_set = local_fs_sets
    end
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
