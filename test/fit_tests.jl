
@testset "umap tests" begin
    using UMAP: fit, transform

    @testset "fit function" begin
        @testset "argument validation tests" begin
            @test_throws ArgumentError fit([1. 1.]; n_neighbors=0) # n_neighbors error
            @test_throws ArgumentError fit([1. 1.], 0; n_neighbors=1) # n_comps error
            @test_throws ArgumentError fit([1. 1.], 2; n_neighbors=1) # n_comps error
            @test_throws ArgumentError fit([1. 1.; 1. 1.; 1. 1.];
                    n_neighbors=1, min_dist = 0.) # min_dist error
        end
    end

    @testset "input type stability tests" begin
        data = rand(5, 100)
        result = fit(data; init=UMAP.UniformInitialization())
        @test result isa UMAPResult
        @test size(result.graph) == (100, 100)
        @test size(result.embedding) == (2, 100)
        @test result.data === data
        @inferred fit(data; init=UMAP.UniformInitialization())

        data = rand(Float32, 5, 100)
        result = fit(data; init=UMAP.UniformInitialization())
        @test result isa UMAPResult
        @test eltype(result.embedding) == Float32
        @inferred fit(data; init=UMAP.UniformInitialization())
    end


    @testset "umap transform" begin
        @testset "argument validation tests" begin
            data = rand(5, 10)
            result = fit(data, 2; n_neighbors=2, n_epochs=1)
            query = rand(5, 8)
            @test_throws ArgumentError transform(result, rand(6, 8); n_neighbors=3) # query size error
            @test_throws ArgumentError transform(result, query; n_neighbors=0) # n_neighbors error
            @test_throws ArgumentError transform(result, query; n_neighbors=15) # n_neighbors error
            @test_throws ArgumentError transform(result, query; n_neighbors=1, min_dist = 0.) # min_dist error
        end

        @testset "transform test" begin
            data = rand(5, 30)
            result = fit(data, 2; n_neighbors=2, n_epochs=1)
            transform_result = transform(result, rand(5, 10); n_epochs=5, n_neighbors=5)
            @test size(transform_result.embedding) == (2, 10)
            @test typeof(transform_result.embedding) == typeof(result.embedding)

            data = rand(Float32, 5, 30)
            result = fit(data, 2; n_neighbors=2, n_epochs=1)
            transform_result = @inferred transform(result, rand(Float32, 5, 50); n_epochs=5, n_neighbors=5)
            @test size(transform_result.embedding) == (2, 50)
            @test typeof(transform_result.embedding) == typeof(result.embedding)
        end
    end
end
