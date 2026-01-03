### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f5d0fb68-3cc8-41c1-9775-a7bdb4e3aac1
let
	docs_dir = dirname(dirname(@__DIR__))
	pkg_dir = dirname(docs_dir)
	
	using Pkg: Pkg
	Pkg.activate(docs_dir)
	Pkg.develop(; path=pkg_dir)
	Pkg.instantiate()
end; # hideall

# ╔═╡ ac958851-2b12-4f08-85aa-36193d43276e
using UMAP

# ╔═╡ d1b241bc-2f6b-4853-b657-b5acbb8914e7
using Distances

# ╔═╡ ff3be6f0-f809-42b8-9457-2542e6002a2e
using StringDistances

# ╔═╡ 279cd6ba-398c-11eb-3726-017ceb9dea5c
md"""
# Advanced Usage
"""

# ╔═╡ 72b19124-3996-11eb-37cb-4184976b0d9b
md"""
## Algorithm
At a high level, the UMAP algorithm proceeds in the following steps:

```julia
knns_dists = knn_search(data, knn_params)
umap_graphs = fuzzy_simplicial_set(knns_dists, knn_params, src_view_params)
umap_graph = coalesce_views(umap_graphs, src_global_params)
embedding = initialize_embedding(umap_graph, tgt_params)
optimize_embedding!(embedding, umap_graph, tgt_params, opt_params)
```
"""

# ╔═╡ 2e7552d4-398c-11eb-2f64-63b8af73b208
md"""
## KNN Search
In a typical workflow, the first step of the UMAP algorithm is to find a (approximate) k-nearest neighbor graph. 
"""

# ╔═╡ ecee2216-398c-11eb-0903-4d55ae073c58
md"""
### Approximate neighbors for vector data
A very simple example of this is to find 4 approximate nearest neighbors for vectors in ``R^n`` using the Euclidean metric:
"""

# ╔═╡ 8f641ed6-398c-11eb-1678-1d25fec1110e
xs = [rand(10) for _ in 1:10];

# ╔═╡ f723f1d8-9669-4bd9-94c3-4d786e45ef68
Docs.doc(UMAP.DescentNeighbors) # hideall

# ╔═╡ d8424a74-398c-11eb-2caa-07c75477d11e
knn_params = UMAP.DescentNeighbors(4, Euclidean());

# ╔═╡ c1410072-398c-11eb-1398-47403e535012
UMAP.knn_search(xs, knn_params)

# ╔═╡ 9279de48-398d-11eb-1e07-1136141af11e
md"""
The return result in this case is a tuple of 4x10 (`n_neighbors` x `n_points`) matrices, one for the indices of the nearest neighbors and the second for the distances.

e.g. `knn_search(xs, knn_params) -> indices, distances`
"""

# ╔═╡ be4537e8-398d-11eb-22ba-b51e1aa3dee8
md"""
The knn parameter struct `DescentNeighbors` uses `NearestNeighborDescent.jl` to find the approximate knns of the data. It also allows passing keyword arguments to `nndescent`:
"""

# ╔═╡ 0862883a-398e-11eb-3188-255fe8d4d14f
knn_params_kw = UMAP.DescentNeighbors(4, Euclidean(), (max_iters=15,));

# ╔═╡ 3067ce2e-398e-11eb-3cde-97c95b306cef
UMAP.knn_search(xs, knn_params_kw)

# ╔═╡ 469b60a4-398e-11eb-13a2-4dab74b0f1bb
md"""
### Precomputed distances
Alternatively, a precomputed distance matrix can be passed in if the pairwise distances are already known. This is done by using the `PrecomputedNeighbors` knn parameter struct (note that `n_neighbors` is still required in order to later construct the fuzzy simplicial set, and for transforming new data):
"""

# ╔═╡ cd0c3398-398e-11eb-3486-cdf107f27159
distances = [0. 2 1;
			 2 0 3;
			 1 3 0];

# ╔═╡ b6547e17-ec79-441e-ae32-48b767bdf731
Docs.doc(UMAP.PrecomputedNeighbors) # hideall

# ╔═╡ 89729294-398e-11eb-2d30-fbed1c13ce51
knn_params_pre = UMAP.PrecomputedNeighbors(2, distances);

# ╔═╡ f3d66db6-398e-11eb-2432-8d2a4804d4c5
UMAP.knn_search(nothing, knn_params_pre)

# ╔═╡ 143aac84-398f-11eb-29f4-05727bb571de
md"""
### Multiple views
One key feature of UMAP is combining multiple, heterogeneous views of the same dataset. For the knn search step, this is set up by passing a `NamedTuple` of data views and a corresponding named tuple of knn parameter structs. The `knn_search` function then broadcasts for each (data, knn_param) pair and returns a named tuple of (indices, distances) that similarly corresponds to the input.

For example, in addition to the vector data `xs` we might also have string data:
"""

# ╔═╡ 16633022-3990-11eb-1e5a-7f96e5fca442
xs_str = [join(rand('A':'Z', 10), "") for _ in 1:10];

# ╔═╡ 650f8432-3990-11eb-0217-37f490ec414c
knn_params_str = UMAP.DescentNeighbors(3, RatcliffObershelp());

# ╔═╡ aeef6b0e-398f-11eb-3aac-d1d6dc7210d5
data_views = (view_1=xs, 
			  view_2=xs_str);

# ╔═╡ a92040d0-3990-11eb-063b-ed48334893db
knn_params_views = (view_1=knn_params, 
				    view_2=knn_params_str);

# ╔═╡ c81b8454-3990-11eb-3376-9faac9aa5987
UMAP.knn_search(data_views, knn_params_views)

# ╔═╡ 1f56f788-e4df-4cbe-bf18-1c3082ec47db
md"""
This is relatively flexible, with one major restriction that the size of the data in each view must be the same. You can even combine `DescentNeighbors` with `PrecomputedNeighbors`.
"""

# ╔═╡ 6cac2bc2-3991-11eb-098b-15e21571c561
md"""
## Fuzzy Simplicial Sets
Once we have one or more set of knns for our data (one for each view), we can construct a global fuzzy simplicial set. This is done via the function

`fuzzy_simplicial_set(...) -> umap_graph::SparseMatrixCSC`

A global fuzzy simplicial set is constructed **for each view** of the data with construction paramaterized by the `SourceViewParams` struct. If there is more than one view, their results are combined to return a single fuzzy simplicial set (represented as a weighted, undirected graph).
"""

# ╔═╡ 0009ca40-3993-11eb-3b0b-db3511e3d9a7
md"""
### Fuzzy simplicial set - one view
To create a fuzzy simplicial set for our original dataset of vectors:
"""

# ╔═╡ 11d94260-b19f-45db-9f3e-a5fba68087d3
Docs.doc(UMAP.SourceViewParams) # hideall

# ╔═╡ 32f95f38-3993-11eb-38e5-216e944d198e
src_view_params = UMAP.SourceViewParams(1, 1, 1);

# ╔═╡ 49d1c3c6-3993-11eb-2e2a-c7c5d3a1de18
knns_dists = UMAP.knn_search(xs, knn_params);

# ╔═╡ 844c0f66-3993-11eb-3a40-992da37a639e
UMAP.fuzzy_simplicial_set(knns_dists, knn_params, src_view_params)

# ╔═╡ fa5d0822-3993-11eb-1152-a5e1333fe70f
md"""
### Fuzzy simplicial set - multiple views
As before, multiple views can be passed to `fuzzy_simplicial_set` - each parameterized by its own `SourceViewParams` - and combined into a single, global fuzzy simplicial set.

Using our combination of vector and string data:
"""

# ╔═╡ 307f647c-3994-11eb-042d-47ae4e4779d2
knns_dists_views = UMAP.knn_search(data_views, knn_params_views);

# ╔═╡ 554503a2-3994-11eb-032d-9122491e1d55
src_view_params_views = (view_1=src_view_params, 
					     view_2=src_view_params);

# ╔═╡ 83b5aa0c-3994-11eb-1ede-7f3440865586
fsset_views = UMAP.fuzzy_simplicial_set(knns_dists_views, knn_params_views, src_view_params_views)

# ╔═╡ 0ce99b28-3995-11eb-08c8-7fe41c8e5ff6
md"""
### Combining views' fuzzy simplicial sets
We need a single umap graph (i.e. global fuzzy simplicial set) in order to perform optimization, so if there are multiple dataset views we must combine their sets.

The views' fuzzy sets are combined via fuzzy set intersection left-to-right (`foldl`), where `mix_ratio` is a weight for the intersection operation, analogous to a weighted mean.
"""

# ╔═╡ 6e982660-482b-41c8-897b-5a2873e7e403
Docs.doc(UMAP.SourceGlobalParams) # hideall

# ╔═╡ 8f21a1b8-3995-11eb-189d-9b0dd552f1c8
src_gbl_params = UMAP.SourceGlobalParams(0.5);

# ╔═╡ fb400814-3995-11eb-0fc8-372701323b2c
graph = UMAP.coalesce_views(fsset_views, src_gbl_params)

# ╔═╡ d2915640-3998-11eb-22c5-adc30c539cd6
md"""
## Initialize and optimize target embedding
Again, the goal of UMAP is to embed points on one manifold onto a new manifold. By default, this is a Euclidean manifold of specified dimension `d`. 

The target embedding is parameterized by a target metric (default `SqEuclidean`, but in principle could be anything) as well as a membership function that computes the fuzzy set membership on the new manifold (the analog to the probabilities contained in the UMAP graph).
"""

# ╔═╡ b8a3f990-2060-4df3-9518-b39b216af95c
Docs.doc(UMAP.MembershipFnParams) # hideall

# ╔═╡ eef78d37-4238-4603-9c63-9fdc301fcf22
Docs.doc(UMAP.TargetParams) # hideall

# ╔═╡ 8cb95f52-3b0d-11eb-209a-c5a6d17a89e7
md"""
## Initialize target embedding
The default target space is d-dimensional Euclidean space, with the squared Euclidean distance metric. Two initialization methods are provided: random and spectral layout.
"""

# ╔═╡ be8795d0-3b0d-11eb-18fd-9f7d61210ae2
md"""
### Uniform random vectors in R^2
"""

# ╔═╡ 011969ec-43aa-11eb-1545-6b63b42277fe
tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold{2}(), SqEuclidean(), UMAP.UniformInitialization(), UMAP.MembershipFnParams(1., 1.));

# ╔═╡ 415b2066-3b0f-11eb-37c7-6fa74b7282b1
umap_graph = UMAP.fuzzy_simplicial_set(knns_dists, knn_params, src_view_params);

# ╔═╡ 880b3960-3b0f-11eb-33a5-a12e8248f6fe
xs_embed = UMAP.initialize_embedding(umap_graph, tgt_params)

# ╔═╡ b7ea70da-43a8-11eb-35b6-d1836e6849c5
md"""
## Optimize target embedding
The embedding is optimized by minimizing the fuzzy set cross entropy loss between the two fuzzy set representations of the data. This is optimization process is parameterized by `OptimizationParams`
"""

# ╔═╡ 56275bb4-25a3-4368-abf4-ce5328f5153e
Docs.doc(UMAP.OptimizationParams) # hideall

# ╔═╡ 0e8db138-43a9-11eb-1dba-dbddcfdd10f7
md"""
### Optimize one epoch
"""

# ╔═╡ a3d69b10-43a9-11eb-1dba-03059f2afcb0
opt_params = UMAP.OptimizationParams(1, 1., 1., 5);

# ╔═╡ d5dad750-63a1-4da6-bfea-cb6aeb2d9fc3
xs_embed_opt = copy(xs_embed);

# ╔═╡ afcb98a0-43a9-11eb-2bc6-cbd18349b749
UMAP.optimize_embedding!(xs_embed_opt, umap_graph, tgt_params, opt_params)

# ╔═╡ 942661db-afd6-42f7-a64d-a8fd42d77053
UMAP.cross_entropy(umap_graph, xs_embed, tgt_params)

# ╔═╡ 95d61f13-a66a-402c-8e55-96f3adec1705
UMAP.cross_entropy(umap_graph, xs_embed_opt, tgt_params)

# ╔═╡ Cell order:
# ╠═f5d0fb68-3cc8-41c1-9775-a7bdb4e3aac1
# ╠═ac958851-2b12-4f08-85aa-36193d43276e
# ╠═d1b241bc-2f6b-4853-b657-b5acbb8914e7
# ╠═ff3be6f0-f809-42b8-9457-2542e6002a2e
# ╟─279cd6ba-398c-11eb-3726-017ceb9dea5c
# ╟─72b19124-3996-11eb-37cb-4184976b0d9b
# ╟─2e7552d4-398c-11eb-2f64-63b8af73b208
# ╟─ecee2216-398c-11eb-0903-4d55ae073c58
# ╠═8f641ed6-398c-11eb-1678-1d25fec1110e
# ╟─f723f1d8-9669-4bd9-94c3-4d786e45ef68
# ╠═d8424a74-398c-11eb-2caa-07c75477d11e
# ╠═c1410072-398c-11eb-1398-47403e535012
# ╟─9279de48-398d-11eb-1e07-1136141af11e
# ╟─be4537e8-398d-11eb-22ba-b51e1aa3dee8
# ╠═0862883a-398e-11eb-3188-255fe8d4d14f
# ╠═3067ce2e-398e-11eb-3cde-97c95b306cef
# ╟─469b60a4-398e-11eb-13a2-4dab74b0f1bb
# ╠═cd0c3398-398e-11eb-3486-cdf107f27159
# ╟─b6547e17-ec79-441e-ae32-48b767bdf731
# ╠═89729294-398e-11eb-2d30-fbed1c13ce51
# ╠═f3d66db6-398e-11eb-2432-8d2a4804d4c5
# ╟─143aac84-398f-11eb-29f4-05727bb571de
# ╠═16633022-3990-11eb-1e5a-7f96e5fca442
# ╠═650f8432-3990-11eb-0217-37f490ec414c
# ╠═aeef6b0e-398f-11eb-3aac-d1d6dc7210d5
# ╠═a92040d0-3990-11eb-063b-ed48334893db
# ╠═c81b8454-3990-11eb-3376-9faac9aa5987
# ╟─1f56f788-e4df-4cbe-bf18-1c3082ec47db
# ╟─6cac2bc2-3991-11eb-098b-15e21571c561
# ╟─0009ca40-3993-11eb-3b0b-db3511e3d9a7
# ╟─11d94260-b19f-45db-9f3e-a5fba68087d3
# ╠═32f95f38-3993-11eb-38e5-216e944d198e
# ╠═49d1c3c6-3993-11eb-2e2a-c7c5d3a1de18
# ╠═844c0f66-3993-11eb-3a40-992da37a639e
# ╟─fa5d0822-3993-11eb-1152-a5e1333fe70f
# ╠═307f647c-3994-11eb-042d-47ae4e4779d2
# ╠═554503a2-3994-11eb-032d-9122491e1d55
# ╠═83b5aa0c-3994-11eb-1ede-7f3440865586
# ╟─0ce99b28-3995-11eb-08c8-7fe41c8e5ff6
# ╟─6e982660-482b-41c8-897b-5a2873e7e403
# ╠═8f21a1b8-3995-11eb-189d-9b0dd552f1c8
# ╠═fb400814-3995-11eb-0fc8-372701323b2c
# ╟─d2915640-3998-11eb-22c5-adc30c539cd6
# ╟─b8a3f990-2060-4df3-9518-b39b216af95c
# ╟─eef78d37-4238-4603-9c63-9fdc301fcf22
# ╟─8cb95f52-3b0d-11eb-209a-c5a6d17a89e7
# ╟─be8795d0-3b0d-11eb-18fd-9f7d61210ae2
# ╠═011969ec-43aa-11eb-1545-6b63b42277fe
# ╠═415b2066-3b0f-11eb-37c7-6fa74b7282b1
# ╠═880b3960-3b0f-11eb-33a5-a12e8248f6fe
# ╟─b7ea70da-43a8-11eb-35b6-d1836e6849c5
# ╟─56275bb4-25a3-4368-abf4-ce5328f5153e
# ╟─0e8db138-43a9-11eb-1dba-dbddcfdd10f7
# ╠═a3d69b10-43a9-11eb-1dba-03059f2afcb0
# ╠═d5dad750-63a1-4da6-bfea-cb6aeb2d9fc3
# ╠═afcb98a0-43a9-11eb-2bc6-cbd18349b749
# ╠═942661db-afd6-42f7-a64d-a8fd42d77053
# ╠═95d61f13-a66a-402c-8e55-96f3adec1705
