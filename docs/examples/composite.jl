### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# ╔═╡ 3525eb82-fd7c-4180-b459-ea25e1fc0642
begin
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ a10735ac-49f3-11eb-0ddb-bf4c5dc7614e
using MLDatasets

# ╔═╡ a669bf1a-49f3-11eb-1eba-1d44761b9fec
using Plots

# ╔═╡ a7852e02-49f3-11eb-2572-6129ad6f7875
using UMAP

# ╔═╡ ebef9f3a-49f5-11eb-044a-7160c1c6eacc
using Distances

# ╔═╡ 6afa3536-49f3-11eb-0f84-b9b08c6c4fd4
md"""
# Composite UMAP
Combining multiple 'views' of the same dataset into a single projection.
"""

# ╔═╡ 87514cf6-49f3-11eb-01ba-4190509cbe58
md"""
As a simple example, we can split the Fashion MNIST dataset in two: one with the top half of the images and the other with the bottom half.
"""

# ╔═╡ ff7bcd50-49f3-11eb-203e-e13e7df76d20
n_points = 20_000

# ╔═╡ b859e40c-49f3-11eb-2b4b-e7f11b310e5a
fmnist_x = FashionMNIST.traintensor(Float64, 1:n_points)

# ╔═╡ f5e80c56-49f4-11eb-2f10-6351ba31e11c
md"""
First, run on full images for baseline.
"""

# ╔═╡ 01a367c0-49f5-11eb-3646-f131e0e6edc9
full_res = UMAP.fit(reshape(fmnist_x, :, n_points))

# ╔═╡ c93d9872-49f3-11eb-2525-a9659a2c89f0
fmnist_x_top = reshape(fmnist_x[1:14, :, :], :, n_points);

# ╔═╡ 1801848c-49f4-11eb-398e-492f459901cb
fmnist_x_bot = reshape(fmnist_x[1:14, :, :], :, n_points);

# ╔═╡ 7a5c13cc-49f4-11eb-2ca8-011f06dde970
fmnist_y = FashionMNIST.trainlabels(1:n_points)

# ╔═╡ 099d2a9e-49f5-11eb-0ca4-9f6832e1601c
scatter(getindex.(full_res.embedding, 1), getindex.(full_res.embedding, 2), color=fmnist_y, legend=false, markersize=3, markerstrokewidth=0.5)

# ╔═╡ 375bcd04-49f4-11eb-3f1c-97917f70ce5c
md"""
Run UMAP on the top and bottom halves separately.
"""

# ╔═╡ 4f5c01f0-49f4-11eb-3af7-4f043c5b5ba4
top_res = UMAP.fit(fmnist_x_top)

# ╔═╡ 6ba47f04-49f4-11eb-3bc3-ed701f349d30
scatter(getindex.(top_res.embedding, 1), getindex.(top_res.embedding, 2), color=fmnist_y, legend=false, markersize=3, markerstrokewidth=0.5)

# ╔═╡ ac782134-49f4-11eb-10ea-c7e52db8b095
bot_res = UMAP.fit(fmnist_x_bot)

# ╔═╡ b1e2d27e-49f4-11eb-3ede-5bb7c13d38c3
scatter(getindex.(bot_res.embedding, 1), getindex.(bot_res.embedding, 2), color=fmnist_y, legend=false, markersize=3, markerstrokewidth=0.5)

# ╔═╡ 4ca6651c-49f5-11eb-2658-07d1d25da74f
md"""
## Pass both top and bottom halves as separate views using the advanced API.
"""

# ╔═╡ 5b92b7ec-49f5-11eb-1307-c12fe103f08a
knn_params = (v1=UMAP.DescentNeighbors(15, Euclidean()),
			  v2=UMAP.DescentNeighbors(15, Euclidean()))

# ╔═╡ 00e92898-49f6-11eb-3d4c-e32da80323a7
src_params = (v1=UMAP.SourceViewParams(1., 1., 1.),
			  v2=UMAP.SourceViewParams(1., 1., 1.))

# ╔═╡ 5c685608-49f6-11eb-3b14-a7946d3f2f88
gbl_params = UMAP.SourceGlobalParams(0.5)

# ╔═╡ 8abf6816-49f6-11eb-06cf-3199216bcb2d
memb_params = UMAP.MembershipFnParams(1., 1.)

# ╔═╡ 9541d0da-49f6-11eb-05c1-955e8ba9b0a3
tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold{2}(), SqEuclidean(), UMAP.SpectralInitialization(), memb_params)

# ╔═╡ b1bb7f2c-49f6-11eb-2709-573d0c09ca37
opt_params = UMAP.OptimizationParams(300, 1., 1., 5)

# ╔═╡ ca29fd90-49f6-11eb-22a6-3b8562305a08
comp_res = UMAP.fit((v1=fmnist_x_top, v2=fmnist_x_bot), knn_params, src_params, gbl_params, tgt_params, opt_params)

# ╔═╡ ff1c5ed0-49f6-11eb-01e0-87ba283ed1d4
scatter(getindex.(comp_res.embedding, 1), getindex.(comp_res.embedding, 2), color=fmnist_y, legend=false, markersize=3, markerstrokewidth=0.5)

# ╔═╡ 2512da86-49f7-11eb-050d-cb01e0be4b7b


# ╔═╡ Cell order:
# ╟─6afa3536-49f3-11eb-0f84-b9b08c6c4fd4
# ╟─87514cf6-49f3-11eb-01ba-4190509cbe58
# ╠═3525eb82-fd7c-4180-b459-ea25e1fc0642
# ╠═a10735ac-49f3-11eb-0ddb-bf4c5dc7614e
# ╠═a669bf1a-49f3-11eb-1eba-1d44761b9fec
# ╠═a7852e02-49f3-11eb-2572-6129ad6f7875
# ╠═ebef9f3a-49f5-11eb-044a-7160c1c6eacc
# ╠═ff7bcd50-49f3-11eb-203e-e13e7df76d20
# ╠═b859e40c-49f3-11eb-2b4b-e7f11b310e5a
# ╟─f5e80c56-49f4-11eb-2f10-6351ba31e11c
# ╠═01a367c0-49f5-11eb-3646-f131e0e6edc9
# ╠═099d2a9e-49f5-11eb-0ca4-9f6832e1601c
# ╠═c93d9872-49f3-11eb-2525-a9659a2c89f0
# ╠═1801848c-49f4-11eb-398e-492f459901cb
# ╠═7a5c13cc-49f4-11eb-2ca8-011f06dde970
# ╟─375bcd04-49f4-11eb-3f1c-97917f70ce5c
# ╠═4f5c01f0-49f4-11eb-3af7-4f043c5b5ba4
# ╠═6ba47f04-49f4-11eb-3bc3-ed701f349d30
# ╠═ac782134-49f4-11eb-10ea-c7e52db8b095
# ╠═b1e2d27e-49f4-11eb-3ede-5bb7c13d38c3
# ╟─4ca6651c-49f5-11eb-2658-07d1d25da74f
# ╠═5b92b7ec-49f5-11eb-1307-c12fe103f08a
# ╠═00e92898-49f6-11eb-3d4c-e32da80323a7
# ╠═5c685608-49f6-11eb-3b14-a7946d3f2f88
# ╠═8abf6816-49f6-11eb-06cf-3199216bcb2d
# ╠═9541d0da-49f6-11eb-05c1-955e8ba9b0a3
# ╠═b1bb7f2c-49f6-11eb-2709-573d0c09ca37
# ╠═ca29fd90-49f6-11eb-22a6-3b8562305a08
# ╠═ff1c5ed0-49f6-11eb-01e0-87ba283ed1d4
# ╠═2512da86-49f7-11eb-050d-cb01e0be4b7b
