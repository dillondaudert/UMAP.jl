
@testset "umap tests" begin
    using UMAP: fit

    @testset "fit function" begin
        @testset "argument validation tests" begin
            @test_throws ArgumentError fit([1. 1.]; n_neighbors=0) # n_neighbors error
            @test_throws ArgumentError fit([1. 1.], 0; n_neighbors=1) # n_comps error
            #@test_throws ArgumentError fit([1. 1.], 2; n_neighbors=1) # n_comps error
            @test_throws ArgumentError fit([1. 1.; 1. 1.; 1. 1.];
                    n_neighbors=1, min_dist = 0.) # min_dist error
        end
    end

    @testset "input type stability tests" begin
        data = rand(5, 100)
        result = fit(data; init=UMAP.UniformInitialization())
        @test result isa UMAP.UMAPResult
        @test size(result.graph) == (100, 100)
        @test size(result.embedding) == (100,)
        @test size(result.embedding[1]) == (2,)
        @test result.data === data
        # TODO: Assess type stability of fit, eventually
        # @inferred fit(data; init=UMAP.UniformInitialization())

        data = rand(Float32, 5, 100)
        result = fit(data; init=UMAP.UniformInitialization())
        @test result isa UMAP.UMAPResult
        # TODO: fix data type stability
        @test_broken eltype(result.embedding) == Vector{Float32}
        # @inferred fit(data; init=UMAP.UniformInitialization())
    end
end
