### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ e17ac940-bdbc-11eb-1f8c-41b46d5552c8
begin
	using UMAP: umap
	using MLDatasets
	using Plots
	using PyCall
end

# ╔═╡ 8e5644c2-219f-41f5-bd57-69a32fe0ea35
md"""
# UMAP for MNIST and FashionMNIST
"""

# ╔═╡ 6686a26a-8f33-4300-8dd2-80890f8138fc
@pyimport umap as py_umap;

# ╔═╡ 19e30f46-bcd1-4403-a986-eb5862c8ffa4
md"""
### 1) MNIST
"""

# ╔═╡ 2873c13b-f583-478a-a183-70a507e06aa2
mnist_x = reshape(MNIST.traintensor(Float64),28*28,:);

# ╔═╡ 4c632fe2-9213-48e7-b6bc-5d2fd1d45ec6
mnist_y = MNIST.trainlabels(1:size(mnist_x, 2));

# ╔═╡ 2eb7cd88-7b04-4ec1-9dca-35ae604c0311
@time res_jl = umap(mnist_x; n_neighbors=10, min_dist=0.001, n_epochs=200);

# ╔═╡ cf976fc2-5563-4f0c-9a53-26a8ca7fad5c
scatter(res_jl[1,:], res_jl[2,:], zcolor=mnist_y, 
        title="MNIST: Julia UMAP", marker=(2, 2, :auto, stroke(0)))

# ╔═╡ b1bfb601-162d-4994-8e18-510ab13bdd22
@time res_py = py_umap.UMAP(n_neighbors=10, min_dist=0.001, n_epochs=200)[:fit_transform](permutedims(mnist_x));

# ╔═╡ 52885e4d-ebb1-4be9-af9c-3de6f30e33d2
scatter(res_py[:,1], res_py[:,2], zcolor=mnist_y, 
        title="MNIST: Python UMAP", marker=(2, 2, :auto, stroke(0)))

# ╔═╡ 8bdd4ed8-4841-4634-a050-913cf08de1aa
md"""
### 2) FashionMNIST
"""

# ╔═╡ 2d98a3cd-3451-47e5-9a9a-9a0dde61dc42
fmnist_x = reshape(FashionMNIST.traintensor(Float64),28*28,:);

# ╔═╡ 9490cc37-fcbc-4b0b-8ba9-7a48f8ac3c3d
fmnist_y = FashionMNIST.trainlabels(1:size(fmnist_x, 2));

# ╔═╡ c441fcd2-334e-4a1c-b476-09ade0274cf5
@time res_jl2 = umap(fmnist_x; n_neighbors=5, min_dist=0.1, n_epochs=200);

# ╔═╡ eecd01b4-3bf2-48a4-8482-f9fc5d08e051
scatter(res_jl2[1,:], res_jl2[2,:], zcolor=fmnist_y, 
        title="FMNIST: Julia UMAP", marker=(2, 2, :auto, stroke(0)))

# ╔═╡ 23358904-df72-4e7e-8b13-c97966407512
@time res_py2 = py_umap.UMAP(n_neighbors=5, min_dist=0.1, n_epochs=200)[:fit_transform](permutedims(fmnist_x));

# ╔═╡ faa1e53d-5d98-46bb-b0af-cac647e96e48
scatter(res_py2[:,1], res_py2[:,2], zcolor=fmnist_y, 
        title="FMNIST: Python UMAP", marker=(2, 2, :auto, stroke(0)))

# ╔═╡ Cell order:
# ╟─8e5644c2-219f-41f5-bd57-69a32fe0ea35
# ╠═e17ac940-bdbc-11eb-1f8c-41b46d5552c8
# ╠═6686a26a-8f33-4300-8dd2-80890f8138fc
# ╟─19e30f46-bcd1-4403-a986-eb5862c8ffa4
# ╠═2873c13b-f583-478a-a183-70a507e06aa2
# ╠═4c632fe2-9213-48e7-b6bc-5d2fd1d45ec6
# ╠═2eb7cd88-7b04-4ec1-9dca-35ae604c0311
# ╠═cf976fc2-5563-4f0c-9a53-26a8ca7fad5c
# ╠═b1bfb601-162d-4994-8e18-510ab13bdd22
# ╠═52885e4d-ebb1-4be9-af9c-3de6f30e33d2
# ╟─8bdd4ed8-4841-4634-a050-913cf08de1aa
# ╠═2d98a3cd-3451-47e5-9a9a-9a0dde61dc42
# ╠═9490cc37-fcbc-4b0b-8ba9-7a48f8ac3c3d
# ╠═c441fcd2-334e-4a1c-b476-09ade0274cf5
# ╠═eecd01b4-3bf2-48a4-8482-f9fc5d08e051
# ╠═23358904-df72-4e7e-8b13-c97966407512
# ╠═faa1e53d-5d98-46bb-b0af-cac647e96e48
