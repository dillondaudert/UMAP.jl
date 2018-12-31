
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
        @test_skip compute_membership_strengths()
    end
    
    @testset "simpl_set_embedding" begin
        @test_skip simpl_set_embedding()
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