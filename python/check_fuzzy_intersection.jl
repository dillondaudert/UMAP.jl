# run on Julia 1.5.3
using PyCall
pyimport_conda("llvmlite", "llvmlite==0.33", "numba") # to try to avoid segfaults by getting Numba with LLVM 9 instead of 10
const py_umap = pyimport_conda("umap", "umap-learn")

# https://github.com/JuliaPy/PyCall.jl/issues/204#issuecomment-192333326
PyCall.PyObject(S::SparseMatrixCSC) =
    pyimport("scipy.sparse")["csc_matrix"]((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))



g1 = sprand(5000,5000,0.01)
g2 = sprand(5000,5000,0.01)

py_ans = py_umap.umap_.general_simplicial_set_intersection(g1, g2, weight=0.5)

jl_ans = UMAP._fuzzy_intersection(nothing, nothing, UMAP.Graph(nothing, g1), UMAP.Graph(nothing, g2); mix_weight=0.5)


rel_error = norm(jl_ans - py_ans.todense()) / max(norm(jl_ans), norm(py_ans.todense()))
