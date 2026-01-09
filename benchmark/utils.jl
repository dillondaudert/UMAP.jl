# utilities for benchmarks

function matrix_data(n_points, n_dims; seed=123456987)
    Random.seed!(seed)
    return randn(n_dims, n_points)
end

function vecvec_data(n_points, n_dims; seed=123456987)
    Random.seed!(seed)
    return [randn(n_dims) for _ in 1:n_points]
end