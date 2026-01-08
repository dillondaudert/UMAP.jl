# UMAP Loss Function

This page describes the fuzzy set cross entropy loss function used in UMAP and how it is optimized during the embedding process.

## Fuzzy Set Cross Entropy

Given two fuzzy simplicial sets, we can consider the 1-skeleta as a fuzzy graph (i.e., a set of edges, where each edge has a probability of existing in the graph). The two sets (of edges) can be compared by computing the set cross entropy.

For a set $A$ and membership functions $\mu: A \to [0, 1]$ and $\nu: A \to [0, 1]$, the set cross entropy is:

```math
C(A, \mu, \nu) = \sum_{a \in A} \left[ \mu(a) \log\frac{\mu(a)}{\nu(a)} + (1 - \mu(a)) \log\frac{1 - \mu(a)}{1 - \nu(a)} \right]
```

In code:
```julia
function cross_entropy(A::Set, μ, ν)
    loss = 0
    for a ∈ A
        loss += μ(a) * log(μ(a) / ν(a)) + (1 - μ(a)) * log((1 - μ(a)) / (1 - ν(a)))
    end
    return loss
end
```

## Generalization to $\ell$-Skeleta

The loss function can be generalized to $\ell$-skeleta by the weighted sum of the set cross entropies of the fuzzy sets of $i$-simplices. That is,

```math
C_\ell(X, Y) = \sum_{i=1}^{\ell} \lambda_i \cdot C(X_i, Y_i)
```

where $X_i$ denotes the $i$-simplices of $X$.

In code:
```julia
function cross_entropy(𝐀::Vector{Set}, μ, ν)
    loss = 0
    for A in 𝐀
        loss += cross_entropy(A, μ, ν)
    end
end
```

## Simplified Loss for Optimization

During optimization, we can simplify the loss function to only consider terms that aren't fixed values and minimize that:

```math
C(A, \mu, \nu) = -\sum_{a \in A} \left[ \mu(a) \log\nu(a) + (1 - \mu(a)) \log(1 - \nu(a)) \right]
```

In code:
```julia
function cross_entropy(A::Set, μ, ν)
    loss = 0
    for a ∈ A
        loss += μ(a) * log(ν(a)) + (1 - μ(a)) * log(1 - ν(a))
    end
    return -loss
end
```

## Stochastic Sampling

Instead of calculating the loss over the entire set (if our set is comprised of the 1-simplices, then calculating this loss would have time complexity $\mathcal{O}(n^2)$), we can sample elements with probability $\mu(a)$ and update according to the value $\nu(a)$. This takes care of the $\mu(a) \log\nu(a)$ term.

For the negative samples, elements are sampled uniformly and assumed to have $\mu(a) = 0$. This results in a sampling distribution of

```math
P(x_i) = \frac{\sum_{a \in A \mid d_0(a) = x_i}(1 - \mu(a))}{\sum_{b \in A \mid d_0(b) \neq x_i}(1 - \mu(b))}
```

which is approximately uniform for sufficiently large datasets.

## Differentiable Membership Function

To optimize this loss with gradient descent, $\nu(v)$ must be differentiable. A smooth approximation for the membership strength of a 1-simplex between two points $x, y$ can be given by the following, with dissimilarity function $\sigma$ and constants $a$, $b$:

```math
\phi(x, y) = \left(1 + a \cdot \sigma(x, y)^{2b}\right)^{-1}
```

In code:
```julia
ϕ(x, y, σ, a, b) = (1 + a*(σ(x, y))^(2b))^(-1)
```

The approximation parameters $a$, $b$ are chosen by non-linear least squares fitting of the following function $\psi$:

```math
\psi(x, y) = \begin{cases}
1 & \text{if } \sigma(x, y) \leq \text{min\_dist} \\
e^{-(\sigma(x, y) - \text{min\_dist})} & \text{otherwise}
\end{cases}
```

In code:
```julia
ψ(x, y, σ, min_dist) = σ(x, y) ≤ min_dist ? 1 : exp(-(σ(x, y) - min_dist))
```

## Optimization Algorithm

Optimizing the embedding is accomplished by stochastic gradient descent, where:
- `fs_set` is the set of $\ell$-simplices (typically 1-simplices)
- `Y_emb` is the target embedding of the points that make up the vertices of `fs_set`
- $\sigma$ is a differentiable distance measure between points in `Y_emb`
- $\phi$ is the differentiable approximation to the fuzzy set membership function for the simplices in the target embedding

The algorithm proceeds as follows:

```julia
function optimize_embedding(fs_set, Y_emb, σ, ϕ, n_epochs, n_neg_samples)
    η = 1  # learning rate
    ∇logϕ(x, y) = gradient((_x, _y) -> log(ϕ(_x, _y, σ)), x, y)
    ∇log1_ϕ(x, y) = gradient((_x, _y) -> log(1 - ϕ(_x, _y, σ)), x, y)

    for e in 1:n_epochs
        for (a, b, p) in fs_set₁  # iterate over 1-simplices
            if rand() ≤ p  # sample with probability p = μ(a)
                # Attractive force (positive sample)
                ∂a, ∂b = η * ∇logϕ(Y_emb[a], Y_emb[b])
                Y_emb[a] -= ∂a

                # Repulsive forces (negative samples)
                for _ in 1:n_neg_samples
                    c = sample(Y_emb)
                    ∂a, ∂c = η * ∇log1_ϕ(Y_emb[a], Y_emb[c])
                    Y_emb[a] -= ∂a
                end
            end
        end
        η = 1 - e/n_epochs  # linear learning rate decay
    end

    return Y_emb
end
```

The algorithm iterates over edges in the fuzzy simplicial set, sampling each edge with probability equal to its membership strength $\mu(a)$. For each sampled edge:

1. **Attractive force**: Apply gradient descent on $\log\phi(x, y)$ to pull connected points together
2. **Repulsive forces**: Sample `n_neg_samples` random points and apply gradient descent on $\log(1 - \phi(x, y))$ to push disconnected points apart

The learning rate $\eta$ decays linearly from 1 to 0 over the course of training.
