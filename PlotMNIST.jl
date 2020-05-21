
using UMAP: umap
using MLDatasets
using Plots
using PyCall
@pyimport umap as py_umap;

mnist_x = MNIST.convert2features(MNIST.traintensor(Float64))
mnist_y = MNIST.trainlabels(1:size(mnist_x, 2));

@time res_jl = umap(mnist_x; n_neighbors=10, min_dist=0.001, n_epochs=200)

scatter(res_jl[1,:], res_jl[2,:], zcolor=mnist_y, 
        title="MNIST: Julia UMAP", marker=(2, 2, :auto, stroke(0)))

@time res_py = py_umap.UMAP(n_neighbors=10, min_dist=0.001, n_epochs=200)[:fit_transform](permutedims(mnist_x))
scatter(res_py[:,1], res_py[:,2], zcolor=mnist_y, 
        title="MNIST: Python UMAP", marker=(2, 2, :auto, stroke(0)))

fmnist_x = FashionMNIST.convert2features(FashionMNIST.traintensor(Float64))
fmnist_y = FashionMNIST.trainlabels(1:size(fmnist_x, 2));

@time res_jl = umap(fmnist_x; n_neighbors=5, min_dist=0.1, n_epochs=200)
x2 = res_jl[1,:]
y2 = res_jl[2,:]
scatter(res_jl[1,:], res_jl[2,:], zcolor=fmnist_y, 
        title="FMNIST: Julia UMAP", marker=(2, 2, :auto, stroke(0)))

@time res_py = py_umap.UMAP(n_neighbors=5, min_dist=0.1, n_epochs=200)[:fit_transform](permutedims(fmnist_x))
scatter(res_py[:,1], res_py[:,2], zcolor=fmnist_y, 
        title="FMNIST: Python UMAP", marker=(2, 2, :auto, stroke(0)))
