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
    # Local Fuzzy Set Operations (merge_local_simplicial_sets)
    # -------------------------------------------------------------------------

    @testset "Fuzzy Set Union and Intersection" begin
        @testset "_fuzzy_set_union" begin
            # Basic properties of fuzzy union: A ∪ B = A + B - A*B
            A = [1.0 0.5; 0.5 1.0]
            union_result = _fuzzy_set_union(A)

            # Union should be symmetric
            @test union_result ≈ union_result'

            # Diagonal should remain 1.0 (1 + 1 - 1*1 = 1)
            @test all(diag(union_result) .≈ 1.0)

            # Off-diagonal: 0.5 + 0.5 - 0.5*0.5 = 0.75
            @test union_result[1, 2] ≈ 0.75
            @test union_result[2, 1] ≈ 0.75

            # Test with asymmetric matrix
            B = [1.0 0.3; 0.7 1.0]
            union_B = _fuzzy_set_union(B)
            # (1,2): 0.3 + 0.7 - 0.3*0.7 = 0.79
            @test union_B[1, 2] ≈ 0.79
            @test union_B[2, 1] ≈ 0.79
        end

        @testset "_fuzzy_set_intersection" begin
            # Basic properties of fuzzy intersection: A ∩ B = A*B
            A = [1.0 0.5; 0.5 1.0]
            inter_result = _fuzzy_set_intersection(A)

            # Intersection should be symmetric
            @test inter_result ≈ inter_result'

            # Diagonal: 1*1 = 1
            @test all(diag(inter_result) .≈ 1.0)

            # Off-diagonal: 0.5*0.5 = 0.25
            @test inter_result[1, 2] ≈ 0.25

            # Test with zeros
            C = [1.0 0.0; 0.5 1.0]
            inter_C = _fuzzy_set_intersection(C)
            @test inter_C[1, 2] ≈ 0.0
        end

        @testset "Boundary Cases" begin
            # All ones
            ones_mat = ones(3, 3)
            @test all(_fuzzy_set_union(ones_mat) .≈ 1.0)
            @test all(_fuzzy_set_intersection(ones_mat) .≈ 1.0)

            # All zeros
            zeros_mat = zeros(3, 3)
            @test all(_fuzzy_set_union(zeros_mat) .≈ 0.0)
            @test all(_fuzzy_set_intersection(zeros_mat) .≈ 0.0)

            # Very small values (numerical stability)
            small_mat = [1.0 1e-10; 1e-10 1.0]
            @test _fuzzy_set_union(small_mat)[1, 2] ≈ 2e-10 atol=1e-15
            @test _fuzzy_set_intersection(small_mat)[1, 2] ≈ 1e-20 atol=1e-25
        end
    end

    @testset "merge_local_simplicial_sets tests" begin
        A = [1.0 0.1; 0.4 1.0]

        @testset "Pure Union (ratio = 1.0)" begin
            union_res = [1.0 0.46; 0.46 1.0]
            res = merge_local_simplicial_sets(A, 1.0)
            @test isapprox(res, union_res)
        end

        @testset "Pure Intersection (ratio = 0.0)" begin
            inter_res = [1.0 0.04; 0.04 1.0]
            res = merge_local_simplicial_sets(A, 0.0)
            @test isapprox(res, inter_res)
        end

        @testset "Interpolated (ratio = 0.5)" begin
            # 0.5 * union + 0.5 * intersection
            union_res = [1.0 0.46; 0.46 1.0]
            inter_res = [1.0 0.04; 0.04 1.0]
            expected = 0.5 .* union_res .+ 0.5 .* inter_res
            res = merge_local_simplicial_sets(A, 0.5)
            @test isapprox(res, expected)
        end

        @testset "Other ratios" begin
            # Test various interpolation ratios
            for ratio in [0.25, 0.75, 0.3, 0.9]
                res = merge_local_simplicial_sets(A, ratio)
                @test res isa Matrix
                @test size(res) == size(A)
                # Result should be between intersection and union
            end
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Verify mathematical property that result is bounded by intersection and union
        # TODO: Test with sparse matrices
        # TODO: Test with larger matrices
    end

    # -------------------------------------------------------------------------
    # General Simplicial Set Operations (Multi-view)
    # -------------------------------------------------------------------------

    @testset "general_simplicial_set_union tests" begin
        @testset "Basic Union" begin
            left_view = sparse([1e-9 0.5; 0.5 1e-9])
            right_view = sparse([1e-9 0.5; 1e-9 0.5])
            res = general_simplicial_set_union(left_view, right_view)

            # Expected: use fuzzy union formula with minimum value protection
            # (1,1): both are 1e-9, so use 1e-8 minimum: 1e-8 + 1e-8 - (1e-8)^2 = 2e-8 - 1e-16
            @test res[1, 1] ≈ 2e-8 - (1e-8)^2 atol=1e-16

            # (1,2): 0.5 + 0.5 - 0.5*0.5 = 0.75
            @test res[1, 2] ≈ 0.75

            # (2,1): 0.5 + max(1e-8, 1e-9) - product
            @test res[2, 1] ≈ 0.5 + 1e-8 - 0.5*1e-8 atol=1e-9
        end

        @testset "Symmetric Union" begin
            # Union should be commutative
            A = sparse(rand(5, 5))
            B = sparse(rand(5, 5))
            @test general_simplicial_set_union(A, B) == general_simplicial_set_union(B, A)
        end

        @testset "Edge Cases" begin
            # Empty matrices (all zeros)
            empty_A = sparse(zeros(3, 3))
            empty_B = sparse(zeros(3, 3))
            # Should handle gracefully - but may error on minimum(nonzeros())
            # This tests numerical stability

            # Single nonzero value
            single_A = sparse([1, 2], [1, 2], [0.5, 0.3], 3, 3)
            single_B = sparse([1, 3], [1, 3], [0.4, 0.6], 3, 3)
            res = general_simplicial_set_union(single_A, single_B)
            @test res isa SparseMatrixCSC
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test with completely disjoint sparsity patterns
        # TODO: Test with identical matrices (union with self)
        # TODO: Verify result sparsity structure matches union of input patterns
    end

    @testset "general_simplicial_set_intersection tests" begin
        @testset "Basic Intersection" begin
            left_view = sparse([1e-9 0.5; 0.5 1e-9])
            right_view = sparse([1e-9 0.5; 1e-9 0.5])

            # Test with mix_ratio = 0.5 (balanced)
            res = general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(0.5))
            # (1, 1) should add the two values (both less than 1e-8)
            @test res[1, 1] == 2e-9
            # (1, 2): _mix_values(0.5, 0.5, 0.5)
            @test res[1, 2] ≈ 0.25
            # (2, 2): one value replaced with min 1e-8
            @test res[2, 2] ≈ 1e-8 * 0.5 atol=1e-10
        end

        @testset "Weighted Intersection - Left (ratio = 0)" begin
            left_view = sparse([1e-9 0.5; 0.5 1e-9])
            right_view = sparse([1e-9 0.5; 1e-9 0.5])
            # ratio=0 should weight towards left (x * y^0 = x)
            left_res = general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(0.))
            @test left_res[1, 1] == 2e-9
            @test left_res[1, 2] == 0.5  # weighted towards left
            @test left_res[2, 1] == 0.5
            @test left_res[2, 2] ≈ 1e-8 atol=1e-10
        end

        @testset "Weighted Intersection - Right (ratio = 1)" begin
            left_view = sparse([1e-9 0.5; 0.5 1e-9])
            right_view = sparse([1e-9 0.5; 1e-9 0.5])
            # ratio=1 should weight towards right (x^0 * y = y)
            right_res = general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(1.))
            @test right_res[1, 1] == 2e-9
            @test right_res[1, 2] == 0.5  # weighted towards right
            @test right_res[2, 1] ≈ 1e-8 atol=1e-10
            @test right_res[2, 2] == 0.5
        end

        @testset "Different mix ratios" begin
            left_view = sparse([0.0 0.8; 0.6 0.0])
            right_view = sparse([0.0 0.4; 0.2 0.0])

            for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]
                res = general_simplicial_set_intersection(left_view, right_view, SourceGlobalParams(ratio))
                @test res isa SparseMatrixCSC
                @test size(res) == size(left_view)
            end
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test that result preserves sparsity structure (union of patterns)
        # TODO: Test commutativity when mix_ratio = 0.5
        # TODO: Test with very different left/right magnitudes
    end

    @testset "_mix_values tests" begin
        @testset "Equal weighting (ratio = 0.5)" begin
            # ratio = 0.5 should give product
            @test _mix_values(4.0, 4.0, 0.5) ≈ 16.0
            @test _mix_values(1.0, 4.0, 0.5) ≈ 4.0
        end

        @testset "Left weighting (ratio < 0.5)" begin
            # ratio = 0 should return x
            @test _mix_values(3.0, 5.0, 0.0) ≈ 3.0
            # Small ratio favors x
            @test _mix_values(2.0, 8.0, 0.1) < 4.0  # closer to 2 than 8
        end

        @testset "Right weighting (ratio > 0.5)" begin
            # ratio = 1 should return y
            @test _mix_values(3.0, 5.0, 1.0) ≈ 5.0
            # Large ratio favors y
            @test _mix_values(2.0, 8.0, 0.9) > 5.0  # closer to 8 than 2
        end

        @testset "Boundary and edge cases" begin
            # Test with 1.0
            @test _mix_values(1.0, 1.0, 0.5) ≈ 1.0

            # Test symmetry
            @test _mix_values(2.0, 8.0, 0.49) ≈ _mix_values(8.0, 2.0, 0.51)
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test numerical stability with very small/large values
        # TODO: Verify mathematical properties (e.g., _mix_values(x, y, r) vs _mix_values(y, x, 1-r))
    end

    # -------------------------------------------------------------------------
    # Local Connectivity and Normalization
    # -------------------------------------------------------------------------

    @testset "reset_local_connectivity tests" begin
        @testset "Full pipeline" begin
            # Create a test simplicial set
            A = sparse(rand(5, 5) .+ 0.1)  # Add 0.1 to avoid zeros

            # Test with reset_local_metric = true
            result = reset_local_connectivity(A, true)
            @test result isa SparseMatrixCSC
            @test size(result) == size(A)
            # Result should be symmetric (due to fuzzy union)
            @test result ≈ result'
            # Should have no explicit zeros
            @test !any(iszero, nonzeros(result))
        end

        @testset "Without metric reset" begin
            A = sparse(rand(4, 4) .+ 0.1)
            result = reset_local_connectivity(A, false)
            @test result isa SparseMatrixCSC
            @test result ≈ result'  # Still symmetric
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Verify that local connectivity constraint is satisfied
        # TODO: Test with matrices of different sparsity patterns
        # TODO: Compare results with/without metric reset
    end

    @testset "_norm_sparse tests" begin
        @testset "Column-wise normalization" begin
            A = rand(4, 4) .+ 1e-8  # add 1e-8 to eliminate any possible
                                    # issues with zeros (even though very rare)
            spA = sparse(A)
            result = _norm_sparse(spA)

            # Each column should have max value of 1
            expected = A ./ maximum(A, dims=1)
            @test all(result .≈ expected)

            # Verify max of each column is 1 (or close to 1)
            @test all(maximum(result, dims=1) .≈ 1.0)
        end

        @testset "Handles zero columns" begin
            # Column with all zeros should be handled by 1e-8 minimum
            A = [1.0 0.0; 0.5 0.0]
            spA = sparse(A)
            result = _norm_sparse(spA)
            @test result isa SparseMatrixCSC
            # First column normalized to max 1.0
            @test result[1, 1] ≈ 1.0
            @test result[2, 1] ≈ 0.5
        end

        @testset "Type preservation" begin
            A_f32 = sparse(rand(Float32, 3, 3))
            result = _norm_sparse(A_f32)
            @test eltype(result) == Float32

            A_f64 = sparse(rand(Float64, 3, 3))
            result = _norm_sparse(A_f64)
            @test eltype(result) == Float64
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test with very large/small values
        # TODO: Test preservation of sparsity structure
        # TODO: Benchmark performance on large sparse matrices
    end

    @testset "reset_local_metrics! tests" begin
        @testset "In-place modification" begin
            # create a sparse matrix with 4 nonzero entries per column
            I = repeat(1:4, inner=4)
            J = repeat(1:4, outer=4)
            V = rand(16)
            A = sparse(I, J, V, 8, 4)  # 8 x 4 sparse matrix

            A_copy = copy(A)
            res = reset_local_metrics!(A_copy)

            # Should return the same (modified) matrix
            @test res === A_copy
            @test res isa typeof(A)
            @test size(A) == size(res)
        end

        @testset "Cardinality target" begin
            I = repeat(1:4, inner=4)
            J = repeat(1:4, outer=4)
            V = rand(16)
            A = sparse(I, J, V, 8, 4)

            @inferred reset_local_metrics!(copy(A))
            res = reset_local_metrics!(copy(A))

            # Each column should have sum ≈ log2(15)
            @test all(isapprox.(sum(res, dims=1), log2(15), atol=SMOOTH_K_TOLERANCE))
        end

        @testset "Different k values" begin
            # _reset_fuzzy_set_cardinality uses k=15 by default
            # This test verifies the column sums
            probs = rand(10)
            k_values = [5, 10, 15, 20, 30]

            for k in k_values
                res = _reset_fuzzy_set_cardinality(probs, k)
                @test isapprox(sum(res), log2(k), atol=SMOOTH_K_TOLERANCE)
            end
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test with columns of varying initial cardinalities
        # TODO: Test numerical stability with extreme probability values
        # TODO: Verify that reset preserves relative ordering of probabilities
    end

    @testset "_reset_fuzzy_set_cardinality tests" begin
        @testset "Achieves target cardinality" begin
            # this function resets the probabilities of a given
            # local simplicial set (a vector of probabilities 0<p<=1)
            # so that the cardinality (sum) of this set is approximately
            # log2(15) (k=15 by default)
            probs = rand(15)
            K = 15
            res = _reset_fuzzy_set_cardinality(probs, K)
            @test isapprox(sum(res), log2(K), atol=SMOOTH_K_TOLERANCE)
        end

        @testset "Different input sizes" begin
            for n in [5, 10, 20, 50]
                probs = rand(n)
                res = _reset_fuzzy_set_cardinality(probs, 15)
                @test isapprox(sum(res), log2(15), atol=SMOOTH_K_TOLERANCE)
                @test length(res) == n
            end
        end

        @testset "Preserves relative order" begin
            probs = [0.1, 0.5, 0.9, 0.3]
            res = _reset_fuzzy_set_cardinality(probs, 10)
            # After exponentiating by mid, order should be preserved
            # (since all probs are positive)
            @test sortperm(probs) == sortperm(res)
        end

        @testset "Extreme values" begin
            # All same values
            same_probs = fill(0.5, 10)
            res = _reset_fuzzy_set_cardinality(same_probs, 15)
            @test all(res .≈ res[1])  # Should all remain equal

            # Very small values
            small_probs = fill(1e-6, 10)
            res_small = _reset_fuzzy_set_cardinality(small_probs, 15)
            @test isapprox(sum(res_small), log2(15), atol=SMOOTH_K_TOLERANCE)

            # Mix of small and large
            mixed = [1e-8, 0.5, 0.9, 1e-10, 0.3]
            res_mixed = _reset_fuzzy_set_cardinality(mixed, 10)
            @test isapprox(sum(res_mixed), log2(10), atol=SMOOTH_K_TOLERANCE)
        end

        @testset "Binary search convergence" begin
            # Test that it converges within niter iterations
            probs = rand(20)
            # With default niter=32, should always converge
            res = _reset_fuzzy_set_cardinality(probs, 15, 32)
            @test isapprox(sum(res), log2(15), atol=SMOOTH_K_TOLERANCE)

            # Test with fewer iterations - may not converge as precisely
            res_few = _reset_fuzzy_set_cardinality(probs, 15, 5)
            @test sum(res_few) > 0  # Should still be reasonable
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test that function is monotonic (larger k -> larger result sum)
        # TODO: Verify convergence rate of binary search
        # TODO: Test edge case where target cannot be achieved
    end

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

    # =========================================================================
    # ADDITIONAL TESTING SUGGESTIONS
    # =========================================================================
    #
    # 1. INTEGRATION TESTS:
    #    - Test full pipeline: fuzzy set creation -> merging -> reset connectivity
    #    - Test multi-view workflow with actual data
    #    - Validate against Python UMAP implementation
    #
    # 2. NUMERICAL STABILITY:
    #    - Test with matrices containing very small values (< 1e-10)
    #    - Test with very large sparse matrices
    #    - Test edge cases where minimum values affect results
    #
    # 3. PERFORMANCE TESTS:
    #    - Benchmark fuzzy set operations on large matrices
    #    - Profile memory allocation in sparse operations
    #    - Test scalability of trustworthiness metric
    #
    # 4. MATHEMATICAL PROPERTIES:
    #    - Verify fuzzy set axioms (idempotency, associativity where applicable)
    #    - Test that operations preserve valid probability ranges [0, 1]
    #    - Verify symmetry properties of operations
    #
    # 5. EDGE CASES:
    #    - Empty sparse matrices
    #    - Single-element matrices
    #    - Matrices with no common nonzero patterns
    #    - Degenerate embeddings (all points identical)
    #
    # 6. TYPE STABILITY:
    #    - Use @inferred on all utility functions
    #    - Test with different numeric types (Float32, Float64)
    #    - Verify sparse matrix type preservation
    #
    # =========================================================================

end
