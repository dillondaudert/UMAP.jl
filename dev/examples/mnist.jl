### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 9f99b293-9ebe-4a92-ac63-e8d148d3dd62
let
	docs_dir = dirname(dirname(@__DIR__))
	pkg_dir = dirname(docs_dir)
	
	using Pkg: Pkg
	Pkg.activate(docs_dir)
	Pkg.develop(; path=pkg_dir)
	Pkg.instantiate()
end; # hideall

# ╔═╡ a0d99c2a-4550-11eb-257b-4dc0fe9c1489
using CairoMakie

# ╔═╡ ad414d14-4550-11eb-30b1-e5144aa81188
begin
	using MLDatasets
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
end

# ╔═╡ b063762a-4550-11eb-3fac-5d67635c5045
using UMAP

# ╔═╡ b48e4f0e-4550-11eb-22ed-4f887d5de4c0
using Distances

# ╔═╡ b795d244-4550-11eb-3a28-0f4afaf888eb
md"""
# UMAP.jl on MNIST Digits
"""

# ╔═╡ 349039ec-4551-11eb-0aaf-0599138a2016
mnist_x = reshape(MNIST.traintensor(Float64), :, 60000);

# ╔═╡ 68589dbe-4551-11eb-3245-751086d9833f
mnist_y = MNIST.trainlabels();

# ╔═╡ 4c9a8bd9-ca61-4f91-93e3-d9397ab7358a
result = UMAP.fit(mnist_x);

# ╔═╡ 87f49caa-456e-11eb-121d-c1d163b2a361
begin
	f = Figure()
	axis = f[1, 1] = Axis(f, title="UMAP.jl - MNIST")
	for d in 0:9
		idx = mnist_y[1:5000] .== d
		scatter!(axis, getindex.(result.embedding[1:5000][idx], 1), getindex.(result.embedding[1:5000][idx], 2), label=string(d), markersize=5)
	end
	f[1, 2] = Legend(f, axis, "Digit", framevisible=false)
	f
end

# ╔═╡ 1b8595b8-498f-11eb-0a85-9980c3f89f3b


# ╔═╡ Cell order:
# ╠═9f99b293-9ebe-4a92-ac63-e8d148d3dd62
# ╠═a0d99c2a-4550-11eb-257b-4dc0fe9c1489
# ╠═ad414d14-4550-11eb-30b1-e5144aa81188
# ╠═b063762a-4550-11eb-3fac-5d67635c5045
# ╠═b48e4f0e-4550-11eb-22ed-4f887d5de4c0
# ╟─b795d244-4550-11eb-3a28-0f4afaf888eb
# ╠═349039ec-4551-11eb-0aaf-0599138a2016
# ╠═68589dbe-4551-11eb-3245-751086d9833f
# ╠═4c9a8bd9-ca61-4f91-93e3-d9397ab7358a
# ╠═87f49caa-456e-11eb-121d-c1d163b2a361
# ╟─1b8595b8-498f-11eb-0a85-9980c3f89f3b
