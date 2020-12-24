### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ a0d99c2a-4550-11eb-257b-4dc0fe9c1489
using Plots

# ╔═╡ ad414d14-4550-11eb-30b1-e5144aa81188
using MLDatasets

# ╔═╡ b063762a-4550-11eb-3fac-5d67635c5045
using UMAP

# ╔═╡ b48e4f0e-4550-11eb-22ed-4f887d5de4c0
using Distances

# ╔═╡ b795d244-4550-11eb-3a28-0f4afaf888eb
md"""
# UMAP.jl on MNIST Digits
"""

# ╔═╡ 349039ec-4551-11eb-0aaf-0599138a2016
mnist_x = reshape(MNIST.traintensor(Float64), :, 60000)

# ╔═╡ 68589dbe-4551-11eb-3245-751086d9833f
mnist_y = MNIST.trainlabels()

# ╔═╡ 73ea981a-4551-11eb-2ed0-2f550f277b00
# run UMAP.fit on subset to compile
UMAP.fit(mnist_x[:, 1:1000]);

# ╔═╡ 917c40e4-4568-11eb-0c04-5f2e0b27d5c1
result = UMAP.fit(mnist_x; metric=CosineDist(), n_neighbors=10, min_dist=0.001, n_epochs=200, neg_sample_rate=5);

# ╔═╡ 87f49caa-456e-11eb-121d-c1d163b2a361
scatter(getindex.(result.embedding[1:1000], 1), getindex.(result.embedding[1:1000], 2), color=mnist_y, label=mnist_y)

# ╔═╡ Cell order:
# ╠═a0d99c2a-4550-11eb-257b-4dc0fe9c1489
# ╠═ad414d14-4550-11eb-30b1-e5144aa81188
# ╠═b063762a-4550-11eb-3fac-5d67635c5045
# ╠═b48e4f0e-4550-11eb-22ed-4f887d5de4c0
# ╟─b795d244-4550-11eb-3a28-0f4afaf888eb
# ╠═349039ec-4551-11eb-0aaf-0599138a2016
# ╠═68589dbe-4551-11eb-3245-751086d9833f
# ╟─73ea981a-4551-11eb-2ed0-2f550f277b00
# ╠═917c40e4-4568-11eb-0c04-5f2e0b27d5c1
# ╠═87f49caa-456e-11eb-121d-c1d163b2a361
