using UMAP: merge_local_simplicial_sets, general_simplicial_set_union, general_simplicial_set_intersection,
            reset_local_connectivity, reset_local_metrics!, _norm_sparse, _reset_fuzzy_set_cardinality,
            _fuzzy_set_union, _fuzzy_set_intersection, _mix_values, trustworthiness, _pairwise_nn,
            SMOOTH_K_TOLERANCE, SourceGlobalParams
using Distances: euclidean, cityblock, sqeuclidean

@testset "utils tests" begin

    # =============================================================================
    # TEST PLAN FOR utils.jl
    # =============================================================================
    #
    # This test suite validates the utility functions for fuzzy set operations,
    # local connectivity management, and embedding evaluation.
    #
    # COVERAGE GOALS:
    # 1. Fuzzy set operations (union, intersection, merging)
    # 2. Multi-view set operations (general_simplicial_set_*)
    # 3. Local connectivity management and normalization
    # 4. Embedding evaluation metrics (trustworthiness)
    # 5. Edge cases and numerical stability
    #
    # TESTING STRATEGY:
    # - Test mathematical properties (e.g., union/intersection commutative)
    # - Test boundary conditions (empty sets, single values, extreme values)
    # - Test numerical stability (very small/large values, zeros)
    # - Validate against known expected values
    # - Test type stability where applicable
    #
    # FUNCTIONS TO TEST:
    # - merge_local_simplicial_sets, _fuzzy_set_union, _fuzzy_set_intersection
    # - general_simplicial_set_union, general_simplicial_set_intersection
    # - _mix_values
    # - reset_local_connectivity, _norm_sparse, reset_local_metrics!
    # - _reset_fuzzy_set_cardinality
    # - trustworthiness, _pairwise_nn
    # =============================================================================


    # -------------------------------------------------------------------------
    # Embedding Evaluation Metrics
    # -------------------------------------------------------------------------

    @testset "trustworthiness tests" begin
        @testset "Perfect embedding" begin
            # If embedding preserves distances exactly, trustworthiness = 1
            X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
            X_embed = [[1.0], [2.0], [3.0], [4.0]]  # 1D embedding preserves order

            trust = trustworthiness(X, X_embed, 2, euclidean)
            @test trust ≈ 1.0
        end

        @testset "Identity embedding" begin
            # Embedding to same space should be perfect
            X = [rand(3) for _ in 1:10]
            trust = trustworthiness(X, X, 5, euclidean)
            @test_broken trust ≈ 1.0
        end

        @testset "Random embedding" begin
            # Random embedding should have lower trustworthiness
            X = [rand(10) for _ in 1:20]
            X_embed = [rand(2) for _ in 1:20]
            trust = trustworthiness(X, X_embed, 5, euclidean)

            # Should be between 0 and 1
            @test 0 ≤ trust ≤ 1
            # Random embedding typically has trust > 0 due to chance
            @test trust > 0
        end

        @testset "Error on dimension mismatch" begin
            X = [rand(3) for _ in 1:10]
            X_embed = [rand(2) for _ in 1:5]  # Different number of points

            @test_throws ErrorException trustworthiness(X, X_embed, 3, euclidean)
        end

        @testset "Different metrics" begin
            X = [rand(5) for _ in 1:15]
            X_embed = [rand(2) for _ in 1:15]

            # Should work with different distance metrics
            trust_euc = trustworthiness(X, X_embed, 4, euclidean)
            trust_sq = trustworthiness(X, X_embed, 4, sqeuclidean)

            @test 0 ≤ trust_euc ≤ 1
            @test 0 ≤ trust_sq ≤ 1
            # Different metrics may give different trustworthiness
        end

        @testset "Different n_neighbors" begin
            X = [rand(4) for _ in 1:20]
            X_embed = [rand(2) for _ in 1:20]

            # Trustworthiness can vary with n_neighbors
            trust_small = trustworthiness(X, X_embed, 3, euclidean)
            trust_large = trustworthiness(X, X_embed, 10, euclidean)

            @test 0 ≤ trust_small ≤ 1
            @test 0 ≤ trust_large ≤ 1
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test with known bad embedding (reversed distances)
        # TODO: Compare with sklearn implementation on same data
        # TODO: Test computational complexity/performance on large datasets
        # TODO: Test with different data distributions (clustered, uniform, etc.)
    end

    @testset "_pairwise_nn tests" begin
        @testset "Basic functionality" begin
            X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]
            nn_ranks = _pairwise_nn(X, euclidean)

            # Should return vector of vectors
            @test length(nn_ranks) == 4
            @test all(isa.(nn_ranks, Vector{Int}))

            # Each vector should have length 4 (one rank per point)
            @test all(length.(nn_ranks) .== 4)

            # Self should always be rank 1 (nearest)
            for i in 1:4
                @test nn_ranks[i][i] == 1
            end
        end

        @testset "Distance ordering" begin
            # Points on a line: [0, 1, 2, 3]
            X = [[0.0], [1.0], [2.0], [3.0]]
            nn_ranks = _pairwise_nn(X, euclidean)

            # For point 0: nearest is 1, then 2, then 3
            @test nn_ranks[1][1] == 1  # self
            @test nn_ranks[1][2] == 2  # next nearest is point at 1
            @test nn_ranks[1][3] == 3  # then point at 2
            @test nn_ranks[1][4] == 4  # farthest is point at 3

            # For point 2 (index 3): nearest neighbors are 2 (self), 1, 3, 0
            @test nn_ranks[3][3] == 1  # self
            # Point at 1 and point at 3 are equidistant
        end

        @testset "Different metrics" begin
            X = [rand(3) for _ in 1:5]

            nn_euc = _pairwise_nn(X, euclidean)
            nn_city = _pairwise_nn(X, cityblock)

            @test length(nn_euc) == 5
            @test length(nn_city) == 5

            # Self should always be rank 1 regardless of metric
            for i in 1:5
                @test nn_euc[i][i] == 1
                @test nn_city[i][i] == 1
            end
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test with duplicate points (how are ties handled?)
        # TODO: Verify that returned ranks are valid permutations
        # TODO: Test memory efficiency for large datasets
    end

end
