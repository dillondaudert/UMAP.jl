using UMAP
using MLDatasets
using PyCall
const py_umap = pyimport_conda("umap", "umap-learn")

include("plotting.jl")

# Supervised
n_points = 10_000
X = reshape(MNIST.traintensor(Float64)[:,:,1:n_points], 28^2, :)
y = MNIST.trainlabels()[1:n_points]

unsup = UMAP_(X; n_neighbors=10, min_dist=0.001, n_epochs=200)
sup = UMAP_(X, y; n_neighbors=10, min_dist=0.001, n_epochs=200, far_dist = 5.0)

plot_umap_comparison((unsup.embedding, y), (sup.embedding,y))

py_unsup = py_umap.UMAP(min_dist =0.001, n_epochs = 200, n_neighbors = 10).fit(permutedims(X))
py_sup = py_umap.UMAP(min_dist =0.001, n_epochs = 200, n_neighbors = 10).fit(permutedims(X), y)

plot_umap_comparison(   (permutedims(py_unsup.embedding_), y),
                        (unsup.embedding,y);
                        titles = ("PyUMAP (unsupervised)", "UMAP.jl (unsupervised)")
                    )

plot_umap_comparison(   (permutedims(py_sup.embedding_), y),
                    (sup.embedding,y);
                    titles = ("PyUMAP (supervised)", "UMAP.jl (supervised)")
                )

                
# Test
n_test_points = 1000
Q = reshape(MNIST.testtensor(Float64)[:,:,1:n_test_points], 28^2, :)
Q_colors = MNIST.testlabels()[1:n_test_points]

unsup_new_embedding = transform(unsup, Q)
sup_new_embedding = transform(sup, Q)

plot_umap_comparison((unsup.embedding, y), (unsup_new_embedding,  Q_colors))

plot_umap_comparison((sup.embedding, y), (sup_new_embedding,  Q_colors))


# plot_umap_comparison((sup.embedding, y), (hcat(sup.embedding, sup_new_embedding), vcat(y, Q_colors)))

