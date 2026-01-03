### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 9139d826-8256-4f6a-9873-bc1a92200057
let
	docs_dir = dirname(dirname(@__DIR__))
	pkg_dir = dirname(docs_dir)
	
	using Pkg: Pkg
	Pkg.activate(docs_dir)
	Pkg.develop(; path=pkg_dir)
	Pkg.instantiate()
end;

# ╔═╡ 26576cce-498e-11eb-3196-8f02f292fa43
using CairoMakie

# ╔═╡ 348af824-498e-11eb-17ee-71b3b2444db8
begin
	using MLDatasets
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
end

# ╔═╡ 367b3ed2-498e-11eb-1b9a-8ba3d8d00cf4
using UMAP

# ╔═╡ 3ea6dea8-4c20-43cb-91c5-eeef34379c94
n_examples = 10_000

# ╔═╡ 378a542a-498e-11eb-22bc-8f6deca7282b
fmnist_x = reshape(FashionMNIST.traintensor(Float64, 1:n_examples), :, n_examples);

# ╔═╡ 4c7f2428-498e-11eb-0052-9f6162c9cfaf
fmnist_y = FashionMNIST.trainlabels(1:n_examples);

# ╔═╡ 9b3cca4a-498e-11eb-3f93-0da77ac72c92
result = UMAP.fit(fmnist_x);

# ╔═╡ b7b05b04-498e-11eb-1419-a55ee447fcfa
begin
	f = Figure()
	axis = f[1, 1] = Axis(f)
	scatter!(axis, getindex.(result.embedding, 1), getindex.(result.embedding, 2), color=fmnist_y, markersize=4)
	f
end

# ╔═╡ f785b85a-4990-11eb-07ac-63e2a5c6ff05
fmnist_x_2 = reshape(FashionMNIST.testtensor(Float64, 1:3000), :, 3000)

# ╔═╡ 78e1a92a-4991-11eb-3e0d-9399e28394c1
fmnist_y_2 = FashionMNIST.testlabels(1:3000)

# ╔═╡ 17b89b5e-4991-11eb-2fbe-e38a81f6bfe0
transform_result = UMAP.transform(result, fmnist_x_2)

# ╔═╡ 63080db0-4991-11eb-2d38-870733985c7c
begin
	f2 = Figure()
	axis2 = f[1, 1] = Axis(f2)
	scatter!(axis2, getindex.(transform_result.embedding, 1), getindex.(transform_result.embedding, 2), color=fmnist_y_2, markersize=4)
	f2
end

# ╔═╡ Cell order:
# ╠═9139d826-8256-4f6a-9873-bc1a92200057
# ╠═26576cce-498e-11eb-3196-8f02f292fa43
# ╠═348af824-498e-11eb-17ee-71b3b2444db8
# ╠═367b3ed2-498e-11eb-1b9a-8ba3d8d00cf4
# ╠═3ea6dea8-4c20-43cb-91c5-eeef34379c94
# ╠═378a542a-498e-11eb-22bc-8f6deca7282b
# ╠═4c7f2428-498e-11eb-0052-9f6162c9cfaf
# ╠═9b3cca4a-498e-11eb-3f93-0da77ac72c92
# ╠═b7b05b04-498e-11eb-1419-a55ee447fcfa
# ╠═f785b85a-4990-11eb-07ac-63e2a5c6ff05
# ╠═78e1a92a-4991-11eb-3e0d-9399e28394c1
# ╠═17b89b5e-4991-11eb-2fbe-e38a81f6bfe0
# ╠═63080db0-4991-11eb-2d38-870733985c7c
