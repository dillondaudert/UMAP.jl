
@testset "all tests" begin
    A = sprand(10000, 10000, 0.001)
    B = dropzeros(A + A' - A .* A')
    
    @testset "constructor" begin
        @testset "argument validation tests" begin
            @test_throws ArgumentError UMAP_([[1.]], 0, 0)
            @test_throws ArgumentError UMAP_([[1.], [1.]], 1, 0)
            @test_throws ArgumentError UMAP_([[1.], [1.]], 1, 2)
            @test_throws ArgumentError UMAP_([[1., 1., 1.], 
                    [1., 1., 1.]], 1, 2, 0.)
        end
        @testset "simple constructor tests" begin
            data = [rand(20) for _ in 1:100]
            k = 5
            umap_struct = UMAP_(data, k, 2)
            @test size(umap_struct.graph) == (100, 100)
            @test issymmetric(umap_struct.graph)
            @test size(umap_struct.embedding) == (2, 100)
        end
    end
    
    @testset "fuzzy_simpl_set" begin
        data = [rand(20) for _ in 1:500]
        k = 5
        umap_graph = fuzzy_simplicial_set(data, k)
        @test issymmetric(umap_graph)
    end
    
    @testset "smooth_knn_dists" begin
        dists = [0., 1., 2., 3., 4., 5.]
        rho = 1
        sigma = smooth_knn_dist(dists, 6, 100, rho, 1e-5)
        psum(ds, r, s) = sum(exp.(-max.(ds .- r, 0.) ./ s))
        @test psum(dists, rho, sigma) - log2(6) < 1e-5
        
        knn_dists = [0. 0. 0.;
                     1. 2. 3.;
                     2. 4. 5.;
                     3. 4. 5.;
                     4. 6. 6.;
                     5. 6. 10.]
        rhos, sigmas = smooth_knn_dists(knn_dists,
                                        6)
        @test rhos == [1., 2., 3.]
        @test all([psum(knn_dists[:,i], rhos[i], sigmas[i]) for i in 1:3] .- log2(6) .< 1e-5)
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
    
    @testset "simplicial_set_embedding" begin
        @test_skip simplicial_set_embedding()
    end
    
    @testset "optimize_embedding" begin
        layout = spectral_layout(B, 5)
        embedding = optimize_embedding(B, layout, 1, 1., 1., 1.)
        @test embedding isa Array{Float64, 2}
    end
    
    @testset "spectral_layout" begin
        layout = spectral_layout(B, 5)
        @test layout isa Array{Float64, 2}
    end
end