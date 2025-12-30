@testset "optimize_embedding tests" begin
    @testset "OptimizationParams tests" begin
        @test_throws ArgumentError UMAP.OptimizationParams(0, .1, .1, 1)
        @test_throws ArgumentError UMAP.OptimizationParams(1, -0.1, 1., 1)
        @test_throws ArgumentError UMAP.OptimizationParams(1, 1., -0.1, 1)
        @test_throws ArgumentError UMAP.OptimizationParams(1, 1., 1., -1)

        params = UMAP.OptimizationParams(1, 1., 1., 1)
        new_params = UMAP.set_lr(params, 0.5)
        @test new_params.lr == 0.5
        @test params.lr == 1.
    end
    @testset "_optimize_embedding!" begin
        # TODO
    end

    @testset "_optimize_embedding! with reference" begin
        # TODO
    end
end