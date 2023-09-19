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

    @testset "general_simplicial_set_union tests" begin
        left_view = sparse([1e-9 0.5; 0.5 1e-9])
        right_view = sparse([1e-9 0.5; 1e-9 0.5])
        res = [2e-8 - (1e-8)^2 0.75; 0.5 + 1e-8 - 0.5*1e-8 0.5 + 1e-8 - 0.5*1e-8]
        @test general_simplicial_set_union(left_view, right_view) == res
    end

    @testset "general_simplicial_set_intersection tests" begin
        left_view = sparse([1e-9 0.5; 0.5 1e-9])
        right_view = sparse([1e-9 0.5; 1e-9 0.5])
        # (1, 1) should add the two values (both less than 1e-8)
        res = [2e-9 0.25; 1e-8*0.5 1e-8*0.5]
        @test general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(0.5)) == res
        # test weighted intersection; left and right
        left_res = [2e-9 0.5; 0.5 1e-8]
        @test general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(0.)) == left_res
        right_res = [2e-9 0.5; 1e-8 0.5]
        @test general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(1.)) == right_res

    end

    @testset "reset_local_connectivity tests" begin

        @testset "_norm_sparse tests" begin
            A = rand(4, 4) .+ 1e-8 # add 1e-8 to eliminate any possible
                                   # issues with zeros (even though very rare)
            spA = sparse(A)
            @test all(A ./ maximum(A, dims=2) .== UMAP._norm_sparse(spA))
        end
    
        @testset "reset_local_metrics! tests" begin
            # create a sparse matrix with 4 nonzero entries per column
            I = repeat(1:4, inner=4)
            J = repeat(1:4, outer=4)
            V = rand(16)
            A = sparse(I, J, V, 8, 4) # 8 x 4 sparse matrix
            @inferred reset_local_metrics!(copy(A))
            res = reset_local_metrics!(copy(A))
            @test res isa typeof(A)
            @test size(A) == size(res)

            @test all(isapprox.(sum(res, dims=1), log2(15), atol=UMAP.SMOOTH_K_TOLERANCE))
    
            @testset "_reset_fuzzy_set_cardinality tests" begin
                # this function resets the probabilities of a given
                # local simplicial set (a vector of probabilities 0<p<=1)
                # so that the cardinality (sum) of this set is approximately
                # log2(15) (k=15 by default)
                probs = rand(15)
                K = 15
                res = UMAP._reset_fuzzy_set_cardinality(probs, K)
                @test isapprox(sum(res), log2(K), atol=UMAP.SMOOTH_K_TOLERANCE)
            end
    
        end
    end

    

end
