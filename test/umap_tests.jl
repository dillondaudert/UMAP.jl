
@testset "all tests" begin
    A = sprand(10000, 10000, 0.001)
    B = dropzeros(A + A' - A .* A')
    
    @testset "constructor" begin
        @test_skip UMAP_()
    end
    
    @testset "local_fuzzy_simpl_set" begin
        @test_skip local_fuzzy_simpl_set()
    end
    
    @testset "smooth_knn_dists" begin
        @test_skip smooth_knn_dists()
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