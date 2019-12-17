
"""
    fuzzy_simplicial_set(data, n_neighbors, metric, local_connectivity, set_op_ratio) -> graph::SparseMatrixCSC

Construct the local fuzzy simplicial sets of each point in `data` by
finding the approximate nearest `n_neighbors`, normalizing the distances
on the local manifolds, and converting the metric spaces to a fuzzy simplicial set.
"""
function fuzzy_simplicial_set(data::AbstractVector,
                              n_neighbors,
                              metric,
                              local_connectivity,
                              set_operation_ratio)

    knns, dists = knn_search(data, n_neighbors, metric)

    sigmas, rhos = smooth_knn_dists(dists, n_neighbors, local_connectivity)

    rows, cols, vals = compute_membership_strengths(knns, dists, sigmas, rhos)
    fuzzy_sets = sparse(rows, cols, vals, length(data), length(data))

    simplicial_set = combine_fuzzy_sets(fuzzy_sets, set_operation_ratio)

    return dropzeros(simplicial_set)
end

@inline function nonzero_dists(dists::AbstractVector{T}) where T
    return dists[dists .> zero(T)]
end

const SMOOTH_K_TOLERANCE = 1e-5

"""
    smooth_knn_dists(dists, k, local_connectivity; <kwargs>) -> knn_dists, nn_dists

Compute the distances to the nearest neighbors for a continuous value `k`. Returns
the approximated distances to the kth nearest neighbor (`knn_dists`)
and the nearest neighbor (nn_dists) from each point.
"""
function smooth_knn_dists(knn_dists::AbstractVector{V},
                          k::Real,
                          local_connectivity::Real;
                          niter::Integer=64,
                          bandwidth::Real=1) where {S, V <: AbstractVector{S}}
    sigmas = Array{S}(undef, length(knn_dists))
    rhos = zeros(S, length(knn_dists))
    for i in 1:length(knn_dists)
        nz_dists = nonzero_dists(knn_dists[i])
        if length(nz_dists) >= local_connectivity
            index = floor(Int, local_connectivity)
            interpolation = local_connectivity - index
            if index > 0
                rhos[i] = nz_dists[index]
                if interpolation > SMOOTH_K_TOLERANCE
                    rhos[i] += interpolation * (nz_dists[index+1] - nz_dists[index])
                end
            else
                rhos[i] = interpolation * nz_dists[1]
            end
        elseif length(nz_dists) > 0
            rhos[i] = maximum(nz_dists)
        end
        sigmas[i] = smooth_knn_dist(knn_dists[i], rhos[i], k, bandwidth, niter)
    end

    return sigmas, rhos
end

# calculate sigma for an individual point
@fastmath function smooth_knn_dist(dists::AbstractVector{T}, rho, k, bandwidth, niter) where {T <: Real}
    target = log2(k)*bandwidth
    lo, mid, hi = zero(T), one(T), typemax(T)
    for n in 1:niter
        psum = sum(exp.((-).(max.(dists .- rho, zero(T)))./mid))
        if abs(psum - target) < SMOOTH_K_TOLERANCE
            break
        end
        if psum > target
            hi = mid
            mid = (lo + hi)/T(2)
        else
            lo = mid
            if hi == typemax(T)
                mid *= T(2)
            else
                mid = (lo + hi) / T(2)
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
function compute_membership_strengths(knns::AbstractVector{V},
                                      dists::AbstractVector,
                                      sigmas::Vector{S},
                                      rhos::Vector{S}) where {T, V <: AbstractVector{T}, S}

    rows = sizehint!(S[], length(knns))
    cols = sizehint!(S[], length(knns))
    vals = sizehint!(T[], length(knns))
    @inbounds for i in 1:length(knns), j in 1:length(knns[i])
        if i == knns[i][j]
            # distance to self always 0
            continue
        else
            d = exp(-max(dists[i][j] - rhos[i], zero(T))/sigmas[i])
            append!(cols, i)
            append!(rows, knns[i][j])
            append!(vals, d)
        end
    end
    return rows, cols, vals
end
