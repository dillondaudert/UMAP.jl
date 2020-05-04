@testset "utils tests" begin
    @testset "knn_search" begin
        @inferred knn_search(rand(5, 100), 5, Euclidean())
        @inferred knn_search(rand(5, 100), 5, Euclidean(), Val(:approximate))
        @testset "pairwise tests" begin
            data = [0. 0. 0.; 0. 1.5 2.]
            true_knns = [2 3 2; 3 1 1]
            true_dists = [1.5 .5 .5; 2. 1.5 2.]
            knns, dists = knn_search(data, 2, Euclidean(), Val(:pairwise))
            @test knns == true_knns
            @test dists == true_dists
        end

        @testset "precomputed tests" begin
            dist_mat = [0. 2. 1.;
                        2. 0. 3.;
                        1. 3. 0.]
            true_knns = [3 1 1;
                         2 3 2]
            true_dists = [1 2 1;
                          2 3 3]
            knns, dists = knn_search(dist_mat, 2, :precomputed)
            @test knns == true_knns
            @test dists == true_dists

        end

        @testset "query data tests" begin
            X = [0.0 -2.0 0.0; 0.0 -1.0 2.0]
            Q = [-2 0; 0 -1]
            true_knns = [2 1; 1 2]
            true_dists = [1.0 1.0; 2.0 2.0]
            knns = [3 1 1; 2 3 2]
            dists = [2.0 2.236 2.0; 2.236 3.606 3.606]
            ret_knns, ret_dists = knn_search(X, Q, 2, Euclidean(), knns, dists)
            @test ret_knns == true_knns
            @test isapprox(ret_dists, true_dists, atol=1e-4)
        end
    end

    @testset "combine_fuzzy_sets tests" begin
        A = [1.0 0.1; 0.4 1.0]
        union_res = [1.0 0.46; 0.46 1.0]
        res = combine_fuzzy_sets(A, 1.0)
        @test isapprox(res, union_res)
        inter_res = [1.0 0.04; 0.04 1.0]
        res = combine_fuzzy_sets(A, 0.0)
        @test isapprox(res, inter_res)

    end

    @testset "fit_ab tests" begin
        @test all((1, 2) .== fit_ab(-1, 0, 1, 2))
    end
end
