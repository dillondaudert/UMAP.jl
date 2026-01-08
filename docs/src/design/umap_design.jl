### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 5b00eeea-d6de-4476-b38d-52a072b8179d
using PlutoUI

# ╔═╡ 65a9ff34-ea53-11f0-a23c-c7b331c1eb4f
md"""
# UMAP.jl
The goals and design of UMAP.jl `v0.2`
"""

# ╔═╡ 6bd39193-cd19-44a3-87a7-1c551baa4d48
md"""
## Overview
1. What is UMAP?
2. UMAP.jl v0.2 Design
3. Future work

### Won't cover:
- Julia
"""

# ╔═╡ f6acc229-3717-4057-87d8-dce4ac78051f
md"""
## What is UMAP?
**Uniform Manifold Approximation and Projection (UMAP)** is an algorithm for transforming data on one (approximated) manifold and projecting it onto another.
- Data visualization, dimensionality reduction, preprocessing, clustering
- Nonlinear dimensionality reduction, similar to t-SNE
- Theoretically motivated, computationally efficient

[McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018](https://arxiv.org/abs/1802.03426)
"""

# ╔═╡ fcb22bdd-5164-4d03-8eee-aa410d132f0a
md"""
### Algorithm
Assumptions:

1. The data is uniformly distributed on a Riemannian manifold;
    - (smooth, evenly sampled data)
2. The Riemannian metric is locally constant;
    - (we can use nearest neighbors to infer local geometry)
3. The manifold is locally connected
    - (nearby points in $R^d$ are also nearby on the manifold)

Given data with some representation, project it into a new **target** manifold

1. Find nearest neighbors
2. Construct fuzzy simplicial sets, i.e. weighted undirected graph
3. Initialize and optimize target embedding via SGD

"""

# ╔═╡ d900e964-3bc3-4b3b-825d-6c82c0b2c7c6
md"""
## UMAP.jl v0.1
- Defined basic `UMAP_` (i.e. fit) and `transform` functions
- Keyword args for parameters
- Only supported a single representation of the data
- Not generic, not well-decomposed
"""

# ╔═╡ 40c3a8e7-16c9-4e36-8b75-5a051b8ff1c5
PlutoUI.LocalResource("./UMAP v0.1.0 Structure.png")

# ╔═╡ 339ac88f-a1fb-4cf0-9b84-6db2a94e5eca
md"""
## UMAP.jl v0.2
"""

# ╔═╡ 87361057-8061-4075-bc5c-f8e6d5b3833a
md"""
### Goals
> Make `UMAP.jl` more **general** and **extensible**.

1. Support **multiple views** of the input data
3. Support future functionality without major refactors:
    1. Supervised views
    2. Different target manifolds and metrics

This motivates our design approach:
> Decompose into pipeline of user-parameterizable stages, extensibility where possible, enabling future features
"""

# ╔═╡ fcc7f870-36a2-4662-8788-b7d2eacd8732
md"""
### User vs. Internal Interface
Written so internal functionality is implemented in a similar way as user-defined extensions, that is via one or both of the following:

1. Define subtypes
2. Implement methods for those subtypes

Then the rest of the algorithm should work as normal.

**Why**?
- Allows contributions to be upstreamed easily
- Implementation doubles as development examples
- "owner as user"; if need to further customize, add features later

Specifically, the `knn_search` and `optimize_embedding` stages
"""

# ╔═╡ 90eef3b9-b4fc-4dae-b1d7-bb7ab3f12dba
md"""
### Example - Multi-view data
Data with multiple "views" has more than one representation. A feature vector and its label, for example.

Another example is a retail product, with text description and image:

> Fitted top in stretch viscose jersey featuring an asymmetric, double-layered front with gathers for a draped effect. Round neckline and long sleeves.
"""

# ╔═╡ 1e5be9d9-a11a-4311-9388-15b00e4a0dbe
PlutoUI.LocalResource("./hm_product.avif", :height=>400)

# ╔═╡ 54377cc9-3a15-4def-897c-5241d0f6d0d4
md"""
## Design
"""

# ╔═╡ 2af73579-0ec5-494f-b6ca-05772388a583
PlutoUI.LocalResource("./UMAP v0.2.0 Structure.png")

# ╔═╡ 77cb976b-6bcf-4ef6-a4ef-fecccefb8180
md"""
```julia
# the generic fit algorithm - works for single and  named tuples of configs
function fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)

    # 1. find (approx) nearest neighbors
    knns_dists = knn_search(data, knn_params)
    # 2. construct the umap graph (global fuzzy simplicial set)
    fs_sets = fuzzy_simplicial_set(knns_dists, knn_params, src_params)
    umap_graph = coalesce_views(fs_sets, gbl_params)
    # 3. initialize the target embedding
    embedding = initialize_embedding(umap_graph, tgt_params)
    # 4. optimize the embedding
    optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)

    config = UMAPConfig(knn_params, src_params, gbl_params, tgt_params, opt_params)
    return UMAPResult(data, embedding, config, knns_dists, fs_sets, umap_graph)
end
```
"""

# ╔═╡ 43c1a69f-02a9-43c3-aeca-19361dd1db07
md"""
## 1. `knn_search`: Find (approximate) nearest neighbors
	knn_search(data, knn_params::NeighborParams) -> knns, dists

- `NeighborParams`: Abstract type, subtypes allow defining the behavior of `knn_search`
    - `DescentNeighbors`: Uses `NearestNeighborDescent.jl` to find approx. knn
    - `PrecomputedNeighbors`: Indicates the user is passing precomputed knn instead

Defining new methods of `knn_search`: 
1. Subtype `T <: NeighborParams`
2. Write a method `knn_search(data, knn_params::T)` that returns the nearest neighbors of each point in the data

Internal design supporting user extension
"""

# ╔═╡ 59eeaa81-dbc3-4066-b09d-b481cf3b392b
md"""
Example, from the implementation:
"""

# ╔═╡ 39ea39a9-4785-4352-9e3c-948ebd0604b0
md"""
```julia
\"\"\"
    knn_search(data, knn_params::DescentNeighbors) -> (knns, dists)

Find approximate nearest neighbors using nndescent.
\"\"\"
function knn_search(
		data, 
		knn_params::DescentNeighbors
)

    knn_graph = NND.nndescent(data, 
							  knn_params.n_neighbors, 
						      knn_params.metric; 
						 	  knn_params.kwargs...)
    return NND.knn_matrices(knn_graph)
end
```
"""

# ╔═╡ 103a5457-ff82-47af-a390-de2602fff132
md"""
**NOTE**: This is the only stage that actually uses the data directly, so supporting precomputed distances or user-defined behavior can extend support to e.g. out-of-core or GPU ANN.
"""

# ╔═╡ b88c7c0e-8935-4669-abeb-b9f247fc2bd2
md"""
Customizing the `knn_search` functionality is as straightforward as subtyping `NeighborParams` and providing an implementation for that subtype of `knn_search`.
"""

# ╔═╡ 73f6b577-ea76-4b5d-939c-718cb99f4653
md"""
## 2. Construct UMAP graph (fuzzy simplicial set)
Calculate the source UMAP graph: a fuzzy set <-> weighted graph.
- The elements are edges $(v, w)$
- Each element has a membership probability (edge weight): $p(v, w)$

For each view:
1. Each point has a local simplicial set, representing the local notion of distance on the manifold (defined via distances to its knn)
2. We combine these via set union/intersection to create the global graph

The graphs of each view are coalesced into a single UMAP graph. This graph is sparse since, for each point we only have non-zero probability to its `k` nearest neighbors. Therefore, it has $O(nk)$ entries.


	fuzzy_simplicial_set((knns, dists), knn_params, src_params::SourceViewParams) -> graph
"""

# ╔═╡ b8184fa9-23b3-475e-9792-3f5b94d06bb6
PlutoUI.LocalResource("./python_umap_graph.png")

# ╔═╡ 9e9ad75e-09bf-4da1-8a55-7cd6bce56b62
md"""
This stage can be parameterized via `SourceViewParams`, passed in by the user

In general **not intended for extensibility**; core, non-generic UMAP algorithm. Mathematical invariants encoded, enforced.
"""

# ╔═╡ cf3ee315-bdd1-4b97-bf2b-3743b5957b18
md"""
## 3. Initialize and Optimize Target Embedding
We embed our data into a target manifold, e.g. Euclidean manifold with associated
metric squared euclidean.

```julia
initialize_embedding(umap_graph, manifold, init) -> embedding
```
Once initialized, we optimize it via stochastic gradient descent. Optimization is parameterized by things like `n_epochs`, `learning_rate`, but also presents an opportunity for optimizing on other manifolds in the future.

```julia
optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)
```

Computationally, this iterates over each non-zero edge in the graph and calculates both attractive and repulsive forces between vertices, as a function of their distance in the target embedding, $O(enkd)$, where $d$ is the dimension of the target embedding and $e$ is the number of epochs.
"""

# ╔═╡ c88ce9cc-3395-4cad-9259-06857c647772
md"""
Internal design to support future extensibility. 

User provides a `Manifold`, some distance `target_metric`, and an initialization method.
- `AbstractInitialization`
    - `UniformInitialization`

- `Manifold`:
    - `_EuclideanManifold{N}`

- `TargetMetric`: Calculate distance between points in `Manifold`, by default is `SqEuclidean`.
"""

# ╔═╡ 97f1834f-61ee-4cdb-b005-353ed50d58f3
md"""
```julia
# randomly initialize in Euclidean space of dimension N
function initialize_embedding(
		umap_graph,                      
		::_EuclideanManifold{N},
        ::UniformInitialization
) where N

    return [20 .* rand(T, N) .- 10 for _ in 1:size(umap_graph, 2)]
end
```
"""

# ╔═╡ ea96e640-00bf-4722-a525-bf0aface148a
md"""
The manifold, initialization, and target metric are all intrinsically connected - grouped into `TargetParams`:

```julia
\"\"\"
    TargetParams{M, D, I, P}(manifold::M, metric::D, init::I, memb_params::P)

Parameters for controlling the target embedding, e.g. the manifold, distance metric, initialization 
method.
\"\"\"
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
```
"""

# ╔═╡ 729b9306-91f7-4f1d-8e75-e15f0eb45c2d
md"""
## Optimization Example
"""

# ╔═╡ 4534dbfa-e1be-4a68-bbfa-fd35fd1793de
PlutoUI.LocalResource("./umap_mnist_2epochs.png")

# ╔═╡ b4e68669-be47-4b6e-94ad-a85dad8fde85
PlutoUI.LocalResource("./umap_mnist_100epochs.png")

# ╔═╡ e731b8d8-ebd6-4ffb-aef5-207ceccee1f1
md"""
## How do we handle multiple "views"?
UMAP assumes the data lies on a kind of Riemannian manifold, which we eventually represent as a weight, undirected graph.

Can construct this for different representations ("views") of the data, each with their own notion of distance, e.g.:
- Vectors in $R^n$ with Euclidean distance (i.e. images)
- Strings with Levenshtein distance
- Categories or labels
"""

# ╔═╡ 176e7d3d-fb0b-4081-9fb0-90f2e499f572
md"""
## v0.2 User API
Support varying levels of usage, basic and advanced.

The basic usage is meant to be user-friendly, with reasonable defaults and opt-in customization.
```julia
	# embed single view into Euclidean space with dim `n_components`,
	# controlling execution via keyword argments
	UMAP.fit(data, n_components; kwargs...)
```

The advanced usage allows complete customization of each stage, multiple data views, etc.
```julia
	UMAP.fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)
```

### Notes on advanced usage
More user-facing frontends can be defined depending on usage and feedback. There are also helper functions for constructing the configuration structs:

```julia
	create_view_config(...)
	create_config(...)
```
"""

# ╔═╡ 05b5b944-403a-42ee-b574-24a237094335
md"""
See the documentation for an example notebook demonstrating the
[advanced usage](https://dillondaudert.github.io/UMAP.jl/dev/examples/advanced_usage/)
"""

# ╔═╡ 84f45f3e-473c-4cd3-a5e2-f94c0f477446
md"""
## Future Work

### Arbitrary target manifolds
Supports defining methods for new combination of types, so the process would be:
1. Use the existing `ManifoldsBase.jl` package for manifold support
    1. Gives us the tools for representing points in manifolds, calculating distances, gradients, tangents, retractions
    2. Write new `initialize_embedding` and `optimize_embedding` methods to handle manifolds generically
"""

# ╔═╡ 7fa76c12-acc8-4542-994c-152050e7bbad
md"""
## (extra) Performance Considerations

### Memory usage
The current design assumes that the UMAP graph and target embedding can fit in memory. 
- the UMAP graph (sparse matrix) $O(nk)$
- embedding $O(nd)$ for a embedding dimension of $d$. 

The data itself may be larger than this, as long as the `knn_search` stage supports it properly, e.g. by precomputing nearest neighbors prior to calling into UMAP.

### Computational complexity
In general, the runtime is dominated by the embedding optimization. Each stage has its own general complexity:

- `knn_search`: When using `nndescent`, the empirical runtime is $O(n^{1.14})$
- `fuzzy_simplicial_set`: Iterates over each point and its knn: $O(nk)$
- `optimize_embedding`: For each epoch $e$, we iterate over each point $n$.
    - Each neighbor with membership probability, fixed to be $\log_2 k$
    - For each point, we also sample $\gamma$ points at random for the repulsive force 
    - For each attractive and repulsive edge force, we calculate a distance as a function of the target embedding $d$
Giving a complexity of about $$O(e \cdot n \cdot \log_2 k \cdot \gamma \cdot d)$$
"""

# ╔═╡ 55c505c7-dd57-405b-863a-d0fb0b97292a
md"""
## (extra) Package Ecosystem
UMAP.jl composes well with existing ecosystem:
1. `NearestNeighborDescent.jl` (by me)
    - Approximate nearest neighbor search algorithm
2. `Distances.jl`
    - Types and implementations for various distance metrics; `SemiMetric`, `Metric`.
    - The invariants represented by these types are used to control behavior

Compatibility within the ecosystem is controlled by versioning according to `semver`.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.77"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.3"
manifest_format = "2.0"
project_hash = "1d704a697008c87b43d48cdb31c6b6d7034fd046"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "6ed167db158c7c1031abf3bd67f8e689c8bdf2b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.77"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═5b00eeea-d6de-4476-b38d-52a072b8179d
# ╟─65a9ff34-ea53-11f0-a23c-c7b331c1eb4f
# ╟─6bd39193-cd19-44a3-87a7-1c551baa4d48
# ╟─f6acc229-3717-4057-87d8-dce4ac78051f
# ╟─fcb22bdd-5164-4d03-8eee-aa410d132f0a
# ╟─d900e964-3bc3-4b3b-825d-6c82c0b2c7c6
# ╟─40c3a8e7-16c9-4e36-8b75-5a051b8ff1c5
# ╟─339ac88f-a1fb-4cf0-9b84-6db2a94e5eca
# ╟─87361057-8061-4075-bc5c-f8e6d5b3833a
# ╟─fcc7f870-36a2-4662-8788-b7d2eacd8732
# ╟─90eef3b9-b4fc-4dae-b1d7-bb7ab3f12dba
# ╟─1e5be9d9-a11a-4311-9388-15b00e4a0dbe
# ╟─54377cc9-3a15-4def-897c-5241d0f6d0d4
# ╟─2af73579-0ec5-494f-b6ca-05772388a583
# ╟─77cb976b-6bcf-4ef6-a4ef-fecccefb8180
# ╟─43c1a69f-02a9-43c3-aeca-19361dd1db07
# ╟─59eeaa81-dbc3-4066-b09d-b481cf3b392b
# ╟─39ea39a9-4785-4352-9e3c-948ebd0604b0
# ╟─103a5457-ff82-47af-a390-de2602fff132
# ╟─b88c7c0e-8935-4669-abeb-b9f247fc2bd2
# ╟─73f6b577-ea76-4b5d-939c-718cb99f4653
# ╟─b8184fa9-23b3-475e-9792-3f5b94d06bb6
# ╟─9e9ad75e-09bf-4da1-8a55-7cd6bce56b62
# ╟─cf3ee315-bdd1-4b97-bf2b-3743b5957b18
# ╟─c88ce9cc-3395-4cad-9259-06857c647772
# ╟─97f1834f-61ee-4cdb-b005-353ed50d58f3
# ╟─ea96e640-00bf-4722-a525-bf0aface148a
# ╟─729b9306-91f7-4f1d-8e75-e15f0eb45c2d
# ╟─4534dbfa-e1be-4a68-bbfa-fd35fd1793de
# ╟─b4e68669-be47-4b6e-94ad-a85dad8fde85
# ╟─e731b8d8-ebd6-4ffb-aef5-207ceccee1f1
# ╟─176e7d3d-fb0b-4081-9fb0-90f2e499f572
# ╟─05b5b944-403a-42ee-b574-24a237094335
# ╟─84f45f3e-473c-4cd3-a5e2-f94c0f477446
# ╟─7fa76c12-acc8-4542-994c-152050e7bbad
# ╟─55c505c7-dd57-405b-863a-d0fb0b97292a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
