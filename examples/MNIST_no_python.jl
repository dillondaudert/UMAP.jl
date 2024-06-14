using UMAP
using MLDatasets
using Distances

include("plotting.jl")

# First, let's get a small sample of the MNIST data.
n_points = 1_000
X = reshape(MNIST.traintensor(Float64)[:, :, 1:n_points], 28^2, :)

# We'll also get the labels
y = MNIST.trainlabels()[1:n_points]

# Now we run UMAP super and unsupervised, via PyCall and via UMAP.jl

## try to match py_umap's number of nndescent iterations
nndescent_kwargs = (max_iters = max(5, round(Int, log2(n_points))), sample_rate = 1)
n_neighbors = 20
n_neighbors_meta = 200

unsup = Embedding(DataView(X; metric=Euclidean(), n_neighbors, nndescent_kwargs);  min_dist=0.001, n_epochs=200)

sup = Embedding(DataView(X; metric=Euclidean(), n_neighbors, nndescent_kwargs), DataView(y;metric=Categorical(far_dist=5.0), n_neighbors=n_neighbors_meta, nndescent_kwargs);  min_dist=0.001, n_epochs=200)

# Bug in NearestNeighbors or Distances?
# NN calls `result_type(metric, data[1], data[2])` instead of `result_type(metric, typeof(data[1]), typeof(data[2])`.
# But this works for arrays...
Distances.result_type(M, ::Int, ::Int) = Distances.result_type(M, Int, Int)
#
let
    scene = plot_umap_comparison((unsup.embedding, y), (sup.embedding, y);
        titles=("UMAP.jl (unsupervised)", "UMAP.jl (supervised)"), markersize=5px)
    save("MNIST_jl_unsup_vs_supervised.png", scene, px_per_unit=3, resolution=(1440, 810))
end



for mix_weight in (0.001, 0.25, 0.5, 0.75, 0.999), n_neighbors_meta in (20, 200)
    sup_cts = Embedding(DataView(X; metric=Euclidean(), n_neighbors, nndescent_kwargs), DataView(y;metric=Euclidean(), n_neighbors=n_neighbors_meta, nndescent_kwargs);  min_dist=0.001, n_epochs=200, mix_weight)

    scene = plot_umap_comparison((sup.embedding, y), (sup_cts.embedding, y);
        titles=("UMAP.jl (supervised; categorical)", "UMAP.jl (supervised; continuous) $(n_neighbors_meta) neighbors), mix_weight=$(mix_weight)"), markersize=5px)
    save("MNIST_jl_vs_jl_supervised_cat_vs_cts_max_weight_$(mix_weight)_neighbors_$(n_neighbors_meta).png", scene, px_per_unit=3, resolution=(1440, 810))


    scene = plot_umap_comparison((unsup.embedding, y), (sup_cts.embedding, y);
    titles=("UMAP.jl (unsupervised)", "UMAP.jl (supervised; continuous) $(n_neighbors_meta) neighbors), mix_weight=$(mix_weight)"), markersize=5px)
save("MNIST_jl_vs_jl_unsupervised_vs_cts_max_weight_$(mix_weight)_neighbors_$(n_neighbors_meta).png", scene, px_per_unit=3, resolution=(1440, 810))

end
