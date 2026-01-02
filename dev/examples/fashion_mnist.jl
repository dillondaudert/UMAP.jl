### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 26576cce-498e-11eb-3196-8f02f292fa43
using Plots

# ╔═╡ 348af824-498e-11eb-17ee-71b3b2444db8
using MLDatasets

# ╔═╡ 367b3ed2-498e-11eb-1b9a-8ba3d8d00cf4
using UMAP

# ╔═╡ 378a542a-498e-11eb-22bc-8f6deca7282b
fmnist_x = reshape(FashionMNIST.traintensor(Float64), :, 60000)

# ╔═╡ 4c7f2428-498e-11eb-0052-9f6162c9cfaf
fmnist_y = FashionMNIST.trainlabels()

# ╔═╡ 9b3cca4a-498e-11eb-3f93-0da77ac72c92
result = UMAP.fit(collect(eachcol(fmnist_x)); n_neighbors=20, min_dist=0.005, n_epochs=200)

# ╔═╡ b7b05b04-498e-11eb-1419-a55ee447fcfa
scatter(getindex.(result.embedding, 1), getindex.(result.embedding, 2), color=fmnist_y, legend=false, markersize=3, markerstrokewidth=0.5)

# ╔═╡ f785b85a-4990-11eb-07ac-63e2a5c6ff05
fmnist_x_2 = reshape(FashionMNIST.testtensor(Float64), :, 10000)

# ╔═╡ 17b89b5e-4991-11eb-2fbe-e38a81f6bfe0
transform_result = UMAP.transform(result, collect(eachcol(fmnist_x_2)), result.config.knn_params, result.config.src_params, result.config.gbl_params, result.config.tgt_params, result.config.opt_params)

# ╔═╡ 78e1a92a-4991-11eb-3e0d-9399e28394c1
fmnist_y_2 = FashionMNIST.testlabels()

# ╔═╡ 63080db0-4991-11eb-2d38-870733985c7c
scatter(getindex.(transform_result.embedding, 1), getindex.(transform_result.embedding, 2), color=fmnist_y_2)

# ╔═╡ Cell order:
# ╠═26576cce-498e-11eb-3196-8f02f292fa43
# ╠═348af824-498e-11eb-17ee-71b3b2444db8
# ╠═367b3ed2-498e-11eb-1b9a-8ba3d8d00cf4
# ╠═378a542a-498e-11eb-22bc-8f6deca7282b
# ╠═4c7f2428-498e-11eb-0052-9f6162c9cfaf
# ╠═9b3cca4a-498e-11eb-3f93-0da77ac72c92
# ╠═b7b05b04-498e-11eb-1419-a55ee447fcfa
# ╠═f785b85a-4990-11eb-07ac-63e2a5c6ff05
# ╠═17b89b5e-4991-11eb-2fbe-e38a81f6bfe0
# ╠═78e1a92a-4991-11eb-3e0d-9399e28394c1
# ╠═63080db0-4991-11eb-2d38-870733985c7c
