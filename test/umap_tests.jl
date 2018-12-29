
@testset "unimplemented tests" begin
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
        @test_skip optimize_embedding()
    end
    
    @testset "spectral_layout" begin
        @test_skip spectral_layout()
    end
end