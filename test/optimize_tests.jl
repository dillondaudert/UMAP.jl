@testset "optimize_embedding tests" begin
    @testset "optimize_embedding!" begin
        # Test basic optimization functionality
        data = rand(5, 20)
        result = fit(data, 2; n_neighbors=5, n_epochs=1)
        
        # Test that optimization modifies embedding
        original_embedding = deepcopy(result.embedding)
        optimize_embedding!(result.embedding, result.graph, result.config.tgt_params, result.config.opt_params)
        
        # Embedding should be modified after optimization
        @test result.embedding != original_embedding
        @test length(result.embedding) == size(data, 2)
        for i in eachindex(result.embedding)
            @test length(result.embedding[i]) == 2
        end
    end

    @testset "optimize_embedding! with reference" begin
        # TODO: more tests
    end
end