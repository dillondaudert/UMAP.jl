@testset "utils tests" begin

    @testset "merge_local_simplicial_sets tests" begin
        A = [1.0 0.1; 0.4 1.0]
        union_res = [1.0 0.46; 0.46 1.0]
        res = merge_local_simplicial_sets(A, 1.0)
        @test isapprox(res, union_res)
        inter_res = [1.0 0.04; 0.04 1.0]
        res = merge_local_simplicial_sets(A, 0.0)
        @test isapprox(res, inter_res)

    end

    @testset "general_simplicial_set_intersection tests" begin
        @test_skip false
    end

    @testset "general_simplicial_set_union tests" begin
        @test_skip false
    end

    @testset "reset_local_connectivity tests" begin
        @test_skip false
    end

    @testset "_norm_sparse tests" begin
        A = rand(4, 4) .+ 1e-8 # add 1e-8 to eliminate any possible
                               # issues with zeros (even though very rare)
        spA = sparse(A)
        @test all(A ./ maximum(A, dims=1) .== UMAP._norm_sparse(spA))
    end

end
