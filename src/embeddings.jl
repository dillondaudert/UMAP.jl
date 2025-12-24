# initialization of the target embedding points and membership fn

abstract type AbstractInitialization end

struct SpectralInitialization <: AbstractInitialization end
struct UniformInitialization <: AbstractInitialization end

# we dispatch both on the initialization method and the manifold.
# this is very tentative; could be replaced by Manifolds.jl
"""
A simple, singleton type representing Euclidean space with dimension N. Points
in this manifold are N-dimensional vectors.
"""
struct _EuclideanManifold{N} end
_EuclideanManifold(N::Integer) = _EuclideanManifold{N}()


"""
    TargetParams{M, D, I, P}(manifold::M, metric::D, init::I, memb_params::P)

Parameters for controlling the target embedding, e.g. the manifold, distance metric, initialization 
method.
"""
struct TargetParams{M, D, I, P}
    "The target manifold in which to embed the data"
    manifold::M
    "The metric used to compute distances on the target manifold"
    metric::D
    "The method of initialization for points on the target manifold"
    init::I
    "Parameters for the membership function of the target embedding (see MembershipFnParams)"
    memb_params::P
end

"""
    initialize_embedding(umap_graph, tgt_params) -> embedding

Initialize the embedding according to tgt_params. Return a list of points
(e.g. Vectors) representing the initial embedded dataset.
"""
function initialize_embedding end

function initialize_embedding(umap_graph, tgt_params::TargetParams)
    return initialize_embedding(umap_graph, tgt_params.manifold, tgt_params.init)
end

# random initialization
function initialize_embedding(umap_graph::AbstractMatrix{T},
                              ::_EuclideanManifold{N},
                              ::UniformInitialization) where {T, N}
    return [20 .* rand(T, N) .- 10 for _ in 1:size(umap_graph, 2)]
end

# spectral initialization
function initialize_embedding(umap_graph::AbstractMatrix{T},
                              manifold::_EuclideanManifold{N},
                              ::SpectralInitialization) where {T, N}
    local embed
    try
        embed = spectral_layout(umap_graph, N)
        # expand
        expansion = 10 / maximum(embed)
        embed .= (embed .* expansion) .+ (1//10000) .* randn.(T)
        embed = collect(eachcol(embed))
    catch e
        @debug "$e\nError encountered in spectral_layout; defaulting to random initialization"
        embed = initialize_embedding(umap_graph, manifold, UniformInitialization())
    end
    return embed
end

# initialize according to a reference embedding
"""
    initialize_embedding(ref_embedding, umap_graph, tgt_params) -> embedding

Initialize an embedding of points corresponding to the columns of the `umap_graph`, by taking weighted mean of
the columns of `ref_embedding`, where weights are values from the columns of `umap_graph`.
"""
function initialize_embedding(ref_embedding::AbstractMatrix,
                              umap_graph::AbstractMatrix,
                              ::TargetParams)
    embed = (ref_embedding * umap_graph) ./ (sum(umap_graph, dims=1) .+ eps(eltype(ref_embedding)))
    return collect(eachcol(embed))
end

function initialize_embedding(ref_embedding::AbstractVector{V},
                              umap_graph::SparseMatrixCSC{T},
                              ::TargetParams) where {V, T}
    embed = V[]
    for col_ind in axes(umap_graph, 2)
        col = umap_graph[:, col_ind]
        embed_point = sum(ref_embedding[col.nzind] .* col.nzval) / (sum(col) + eps(T))
        push!(embed, embed_point)
    end
    return embed
end


"""
    spectral_layout(graph, embed_dim) -> embedding

Initialize the graph layout with spectral embedding.
"""
function spectral_layout(graph::SparseMatrixCSC{T},
                         embed_dim::Integer) where {T<:Real}
    graph_f64 = convert.(Float64, graph)
    D_ = Diagonal(dropdims(sum(graph_f64; dims=2); dims=2))
    D = inv(sqrt(D_))
    # normalized laplacian
    L = Symmetric(I - D*graph*D)

    k = embed_dim+1
    num_lanczos_vectors = max(2k+1, round(Int, sqrt(size(L, 1))))
    # get the 2nd - embed_dim+1th smallest eigenvectors
    eigenvals, eigenvecs = Arpack.eigs(L; nev=k,
                                   ncv=num_lanczos_vectors,
                                   which=:SM,
                                   tol=1e-4,
                                   v0=ones(Float64, size(L, 1)),
                                   maxiter=size(L, 1)*5)
    layout = permutedims(eigenvecs[:, 2:k])::Array{Float64, 2}
    return convert.(T, layout)
end
