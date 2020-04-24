
@testset "umap tests" begin

    @testset "constructor" begin
        @testset "argument validation tests" begin
            data = rand(5, 10)
            @test_throws ArgumentError UMAP_([1. 1.]; n_neighbors=0) # n_neighbors error
            @test_throws ArgumentError UMAP_([1. 1.], 0; n_neighbors=1) # n_comps error
            @test_throws ArgumentError UMAP_([1. 1.], 2; n_neighbors=1) # n_comps error
            @test_throws ArgumentError UMAP_([1. 1.; 1. 1.; 1. 1.];
                    n_neighbors=1, min_dist = 0.) # min_dist error
        end
    end

    @testset "input type stability tests" begin
        data = rand(5, 100)
        umap_ = UMAP_(data; init=:random)
        @test umap_ isa UMAP_{Float64}
        @test size(umap_.graph) == (100, 100)
        @test size(umap_.embedding) == (2, 100)

        data = rand(Float32, 5, 100)
        @test UMAP_(data; init=:random) isa UMAP_{Float32}
    end

    @testset "fuzzy_simpl_set" begin
        data = rand(20, 500)
        k = 5
        umap_graph = fuzzy_simplicial_set(data, k, Euclidean(), 1, 1.)
        @test issymmetric(umap_graph)
        @test all(0. .<= umap_graph .<= 1.)
        data = rand(Float32, 20, 500)
        umap_graph = fuzzy_simplicial_set(data, k, Euclidean(), 1, 1.f0)
        @test issymmetric(umap_graph)
        @test eltype(umap_graph) == Float32

        data = 2 .* rand(20, 1000) .- 1
        umap_graph = fuzzy_simplicial_set(data, k, CosineDist(), 1, 1.)
        @test issymmetric(umap_graph)
        @test all(0. .<= umap_graph .<= 1.)
    end

    @testset "smooth_knn_dists" begin
        dists = [0., 1., 2., 3., 4., 5.]
        rho = 1
        k = 6
        local_connectivity = 1
        bandwidth = 1.
        niter = 64
        sigma = smooth_knn_dist(dists, rho, k, bandwidth, niter)
        psum(ds, r, s) = sum(exp.(-max.(ds .- r, 0.) ./ s))
        @test psum(dists, rho, sigma) - log2(k)*bandwidth < SMOOTH_K_TOLERANCE

        knn_dists = [0. 0. 0.;
                     1. 2. 3.;
                     2. 4. 5.;
                     3. 4. 5.;
                     4. 6. 6.;
                     5. 6. 10.]
        rhos, sigmas = smooth_knn_dists(knn_dists, k, local_connectivity)
        @test rhos == [1., 2., 3.]
        diffs = [psum(knn_dists[:,i], rhos[i], sigmas[i]) for i in 1:3] .- log2(6)
        @test all(diffs .< SMOOTH_K_TOLERANCE)

        knn_dists = [0. 0. 0.;
                     0. 1. 2.;
                     0. 2. 3.]
        rhos, sigmas = smooth_knn_dists(knn_dists, 2, 1)
        @test rhos == [0., 1., 2.]

        rhos, sigmas = smooth_knn_dists(knn_dists, 2, 1.5)
        @test rhos == [0., 1.5, 2.5]
    end

    @testset "compute_membership_strengths" begin
        knns = [1 2 3; 2 1 2]
        dists = [0. 0. 0.; 2. 2. 3.]
        rhos = [2., 1., 4.]
        sigmas = [1., 1., 1.]
        true_rows = [1, 2, 2, 1, 3, 2]
        true_cols = [1, 1, 2, 2, 3, 3]
        true_vals = [0., 1., 0., exp(-1.), 0., 1.]
        rows, cols, vals = compute_membership_strengths(knns, dists, rhos, sigmas)
        @test rows == true_rows
        @test cols == true_cols
        @test vals == true_vals
    end

    @testset "optimize_embedding" begin
        Random.seed!(0)
        A = sprand(10000, 10000, 0.001)
        B = dropzeros(A + A' - A .* A')
        layout = initialize_embedding(B, 5, Val(:random))
        n_epochs = 1
        initial_alpha = 1.
        min_dist = 1.
        spread = 1.
        gamma = 1.
        neg_sample_rate = 5
        embedding = optimize_embedding(B, layout, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate)
        @test embedding isa Array{Array{Float64, 1}, 1}
    end

    @testset "spectral_layout" begin
        A = sprand(10000, 10000, 0.001)
        B = dropzeros(A + A' - A .* A')
        layout = spectral_layout(B, 5)
        @test layout isa Array{Float64, 2}
        @inferred spectral_layout(B, 5)
        layout32 = spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
        @test layout32 isa Array{Float32, 2}
        @inferred spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
    end

    @testset "initialize_embedding" begin
        graph = [0.75 1.0 0.5 0.5 0.5 0.25; 
                 1.0 1.33333 0.666667 0.666667 0.666667 0.333333; 
                 0.5 0.666667 0.333333 0.333333 0.333333 0.166667; 
                 0.5 0.666667 0.333333 0.333333 0.333333 0.166667; 
                 0.5 0.666667 0.333333 0.333333 0.333333 0.166667; 
                 0.25 0.333333 0.166667 0.166667 0.166667 0.0833333]
        ref_embedding = [3 4 2;
                         1 5 3]

        embedding = initialize_embedding(graph, ref_embedding, [1,2,3], [4,5,6])
        @test embedding isa Array{Array{Float64, 1}, 1}
        actual = [[3.2, 3.0], [3.2, 3.0], [3.2, 3.0]]
        @test length(embedding) == length(actual)
        for i in 1:length(embedding)
            @test length(embedding[i]) == length(actual[i])
        end
        @test isapprox(embedding, actual, atol=1e-4)

        embedding = initialize_embedding(graph, ref_embedding, [4,5], [2,3,6])
        @test embedding isa Array{Array{Float64, 1}, 1}
        actual = [[3.14286, 2.42857], [3.14286, 2.42857]]
        @test length(embedding) == length(actual)
        for i in 1:length(embedding)
            @test length(embedding[i]) == length(actual[i])
        end
        @test isapprox(embedding, actual, atol=1e-4)
    end

    @testset "optimize_embedding_with_reference" begin
        graph = sparse(Symmetric(sprand(6,6,0.4)))
        embedding = Vector{Float64}[[3, 1], [4, 5], [2, 3], [1, 7], [6, 3], [2, 6]]

        n_epochs = 1
        initial_alpha = 1.
        min_dist = 1.
        spread = 1.
        gamma = 1.
        neg_sample_rate = 5
        query_inds_list = [[1,2,3], [1,3,6]]
        ref_inds_list = [[4,5,6], [2,4,5]]
        for (query_inds, ref_inds) in zip(query_inds_list, ref_inds_list)
            res_embedding = optimize_embedding(graph, embedding, query_inds, ref_inds, n_epochs, initial_alpha, min_dist, spread, gamma, neg_sample_rate, nothing, nothing, move_ref=false)
            @test res_embedding isa Array{Array{Float64, 1}, 1}
            @test length(res_embedding) == length(embedding)
            for i in 1:length(res_embedding)
                @test length(res_embedding[i]) == length(embedding[i])
            end
            @test isapprox(res_embedding[ref_inds], embedding[ref_inds], atol=1e-5)
        end
    end

end
