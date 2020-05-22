using UMAP
using MLDatasets
using PyCall
const py_umap = pyimport_conda("umap", "umap-learn")

include("plotting.jl")

# First, let's get a small sample of the MNIST data.
n_points = 10_000
X = reshape(MNIST.traintensor(Float64)[:, :, 1:n_points], 28^2, :)

# We'll also get the labels
y = MNIST.trainlabels()[1:n_points]

# Now we run UMAP super and unsupervised, via PyCall and via UMAP.jl
# max_iters =  # try to match py_umap's number of nndescent iterations
nndescent_kwargs = (max_iters = max(5, round(Int, log2(n_points))), sample_rate = 1)
unsup = UMAP_(X; n_neighbors=10, min_dist=0.001, n_epochs=200,
              nndescent_kwargs = nndescent_kwargs)
py_unsup = py_umap.UMAP(min_dist=0.001, n_epochs=200, n_neighbors=10).fit(permutedims(X))

scene = plot_umap_comparison((permutedims(py_unsup.embedding_), y), (unsup.embedding, y);
                             titles=("PyUMAP (unsupervised)", "UMAP.jl (unsupervised)"))
save("MNIST_py_vs_jl_unsupervised.png", scene, px_per_unit=3, resolution=(1440, 810))


sup = UMAP_(X, y; n_neighbors=10, min_dist=0.001, n_epochs=200, far_dist=5.0,
        nndescent_kwargs = nndescent_kwargs)
py_sup = py_umap.UMAP(min_dist=0.001, n_epochs=200, n_neighbors=10).fit(permutedims(X), y)

scene = plot_umap_comparison((permutedims(py_sup.embedding_), y), (sup.embedding, y);
                             titles=("PyUMAP (supervised)", "UMAP.jl (supervised)"))
save("MNIST_py_vs_jl_supervised.png", scene, px_per_unit=3, resolution=(1440, 810))


scene = plot_umap_comparison((unsup.embedding, y), (sup.embedding, y);
                             titles=("UMAP.jl (unsupervised)", "UMAP.jl (supervised)"))
save("MNIST_jl_unsup_vs_supervised.png", scene, px_per_unit=3, resolution=(1440, 810))


scene = plot_umap_comparison((permutedims(py_unsup.embedding_), y),
                             (permutedims(py_sup.embedding_), y);
                             titles=("PyUMAP (unsupervised)", "PyUMAP (supervised)"))
save("MNIST_py_unsup_vs_supervised.png", scene, px_per_unit=3, resolution=(1440, 810))
