
    @testset "umap transform tests" begin
        @testset "argument validation tests" begin
            data = rand(5, 100)
            result = UMAP.fit(data, 2; n_neighbors=4, n_epochs=1)
            query = rand(5, 8)
            @inferred UMAP.transform(result, query)
            vec_query = [rand(5) for _ in 1:8]
            # the type of the query must match the original data exactly
            @test_throws MethodError UMAP.transform(result, vec_query)
        end

        @testset "transform test" begin
            data = rand(5, 30)
            result = UMAP.fit(data, 2; n_neighbors=2, n_epochs=1)
            transform_result = UMAP.transform(result, rand(5, 10))
            @test size(transform_result.embedding) == (2, 10)
            @test typeof(transform_result.embedding) == typeof(result.embedding)

            data = rand(Float32, 5, 30)
            result = UMAP.fit(data, 2; n_neighbors=2, n_epochs=1)
            transform_result = UMAP.transform(result, rand(Float32, 5, 10))
            @test size(transform_result.embedding) == (2, 10)
            @test typeof(transform_result.embedding) == typeof(result.embedding)
        end
    end