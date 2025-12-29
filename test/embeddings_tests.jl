@testset "embeddings tests" begin
    @testset "spectral_layout" begin
        A = sprand(1000, 1000, 0.01)
        B = dropzeros(A + A' - A .* A')
        layout = UMAP.spectral_layout(B, 5)
        @test layout isa Array{Float64, 2}
        @test size(layout) == (5, 1000)
        @inferred UMAP.spectral_layout(B, 5)
        
        layout32 = UMAP.spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
        @test layout32 isa Array{Float32, 2}
        @test size(layout32) == (5, 1000)
        @inferred UMAP.spectral_layout(convert(SparseMatrixCSC{Float32}, B), 5)
    end

    @testset "initialize_embedding" begin
        # Test spectral initialization
        graph = sprand(50, 50, 0.1)
        graph = dropzeros(graph + graph')
        
        tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold(2), 
                                     Distances.SqEuclidean(), 
                                     UMAP.SpectralInitialization(),
                                     UMAP.MembershipFnParams(0.1, 1.0))
        
        embedding = UMAP.initialize_embedding(graph, tgt_params)
        @test embedding isa AbstractVector{<:AbstractVector{Float64}}
        @test length(embedding) == size(graph, 2)
        for i in eachindex(embedding)
            @test length(embedding[i]) == 2
        end
        
        # Test uniform initialization
        tgt_params_uniform = UMAP.TargetParams(UMAP._EuclideanManifold(2), 
                                             Distances.SqEuclidean(), 
                                             UMAP.UniformInitialization(),
                                             UMAP.MembershipFnParams(0.1, 1.0))
        
        embedding_uniform = UMAP.initialize_embedding(graph, tgt_params_uniform)
        @test embedding_uniform isa AbstractVector{<:AbstractVector{Float64}}
        @test length(embedding_uniform) == size(graph, 2)
        for i in eachindex(embedding_uniform)
            @test length(embedding_uniform[i]) == 2
            # Check that points are in expected range for uniform init [-10, 10]
            @test all(-10 ≤ x ≤ 10 for x in embedding_uniform[i])
        end
        
        # Test Float32 type stability
        graph32 = convert(SparseMatrixCSC{Float32}, graph)
        tgt_params32 = UMAP.TargetParams(UMAP._EuclideanManifold(2), 
                                       Distances.SqEuclidean(), 
                                       UMAP.UniformInitialization(),
                                       UMAP.MembershipFnParams(0.1f0, 1.0f0))
        
        embedding32 = UMAP.initialize_embedding(graph32, tgt_params32)
        @test embedding32 isa AbstractVector{<:AbstractVector{Float32}}
        @test length(embedding32) == size(graph32, 2)
    end

    @testset "initialize_embedding with reference" begin
        # TODO
    end
end