# notes on the fuzzy set cross entropy loss for UMAP

"""
Given two fuzzy simplicial sets, we can consider the 1-skeleta as
a fuzzy graph (i.e. a set of edges, where each edge has a probability of existing
in the graph). The two sets (of edges) can be compared by computing the
set cross entropy.

For a set A and membership functions μ: A → [0, 1], ν: A → [0, 1], the set cross
entropy is:
"""

function cross_entropy(A::Set, μ, ν)
    loss = 0
    for a ∈ A
        loss += μ(a) * log(μ(a) / ν(a)) + (1 - μ(a)) * log((1 - μ(a)) / (1 - ν(a)))
    end
    return loss
end

"""
The loss function can be generalized to 𝒍-skeleta by the weighted sum of the
set cross entropies of the fuzzy sets of 𝑖-simplices. That is,

C_l(X, Y) = Σ_{i=1}^{l}(λᵢ * C(Xᵢ, Yᵢ)),

where Xᵢ denotes the 𝑖-simplices of X.
"""
function cross_entropy(𝐀::Vector{Set}, μ, ν)
    loss = 0
    for A in 𝐀
        loss += cross_entropy(A, μ, ν)
    end
end

"""
During optimization, we can simplify the loss function to only consider terms
that aren't fixed values and minimize that:

C(A, μ, ν) = - Σ_{a ∈ A} ( μ(a) * log(ν(a)) + (1 - μ(a)) * log(1 - ν(a)))
"""
function cross_entropy(A::Set, μ, ν)
    loss = 0
    for a ∈ A
        loss += μ(a) * log(ν(a)) + (1 - μ(a)) * log(1 - ν(a))
    end
    return -loss
end

"""
Instead of calculating the loss over the entire set (if our set is comprised
of the 1-simplices, then calculating this loss would have time complexity
𝒪(n²)), we can sample elements with probability μ(a) and update according to
the value ν(a). This takes care of the μ(a) * log(ν(a)) term. For the
negative samples, elements are sampled uniformly and assumed to have μ(a) = 0.
This results in a sampling distribution of

P(xᵢ) = Σ_{a ∈ A | d₀(a) = xᵢ}(1 - μ(a)) / Σ_{b ∈ A | d₀(b) ≠ xᵢ}(1 - μ(b)),

which is approximately uniform for sufficiently large datasets.
"""
function sample_distribution(X, A, μ) end

"""
To optimize this loss with gradient descent, ν(v) must be differentiable. A
smooth approximation for the membership strength of a 1-simplex between two
points x, y, can be given by the following, with dissimilarity function `σ`,
and constants `α`, `β`:
"""
ϕ(x, y, σ, α, β) = (1 + α*(σ(x, y))^β)^(-1)

"""
The approximation parameters `α`, `β` are chosen by non-linear least squares
fitting of the following function ψ:
"""
ψ(x, y, σ, min_dist) = 1 if σ(x, y) ≤ min_dist else exp(-(σ(x, y) - min_dist))

"""
Optimizing the embedding is therefore accomplished by the following, where
`fs_set` is the set of 𝒍-simplices (1-simplices most likely), `Y_emb` is the
target embeddings of the points that make up the vertices of `fs_set`, σ is
a differentiable distance measure between points in `Y_emb`, and `ϕ` is the
differentiable approximation to the fuzzy set membership function for the
simplices in the target embedding.
"""
function optimize_embedding(fs_set, Y_emb, σ, ϕ, n_epochs, n_neg_samples)
    η = 1
    ∇logϕ(x, y) = gradient((_x, _y) -> log(ϕ(_x, _y, σ)), x, y)
    ∇log1_ϕ(x, y) = gradient((_x, _y) -> log(1 - ϕ(_x, _y, α)), x, y)
    for e in 1:n_epochs
        for (a, b, p) in fs_set₁ # 1-simplices here
            if rand() ≤ p
                ∂a, ∂b = η * ∇logϕ(Y_emb[a], Y_emb[b])
                Y_emb[a] -= ∂a
                # Y_emb[b] -= ∂b
                for _ in 1:n_neg_samples
                    c = sample(Y_emb)
                    ∂a, ∂c = η * ∇log1_ϕ(Y_emb[a], Y_emb[c])
                    Y_emb[a] -= ∂a
                    # Y_emb[c] -= ∂c
                end
            end
        end
        η = 1 - e/n_epochs
    end
    return Y_emb
end
