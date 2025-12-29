using UMAP: fuzzy_simplicial_set, coalesce_views, smooth_knn_dists, smooth_knn_dist, compute_membership_strengths,
            DescentNeighbors, SourceViewParams, SourceGlobalParams

@testset "simplicial sets tests" begin

    # -------------------------------------------------------------------------
    # SourceViewParams
    # -------------------------------------------------------------------------

    @testset "SourceViewParams" begin
        @testset "Valid Construction" begin
            # Default Float64 construction
            params = UMAP.SourceViewParams(1.0, 1.0, 1.0)
            @test params.set_operation_ratio == 1.0
            @test params.local_connectivity == 1.0
            @test params.bandwidth == 1.0
            @test params isa UMAP.SourceViewParams{Float64}

            # Explicit type parameter
            params_f32 = UMAP.SourceViewParams{Float32}(0.5, 2.0, 1.5)
            @test params_f32 isa UMAP.SourceViewParams{Float32}
            @test params_f32.set_operation_ratio isa Float32

            # Type promotion
            params_mixed = UMAP.SourceViewParams(1, 1.0, 1.5)
            @test params_mixed isa UMAP.SourceViewParams{Float64}
            @test all(isa.([params_mixed.set_operation_ratio,
                           params_mixed.local_connectivity,
                           params_mixed.bandwidth], Float64))
        end

        @testset "Boundary Values" begin
            # set_operation_ratio boundaries
            @test UMAP.SourceViewParams(0.0, 1.0, 1.0).set_operation_ratio == 0.0
            @test UMAP.SourceViewParams(1.0, 1.0, 1.0).set_operation_ratio == 1.0
            @test UMAP.SourceViewParams(0.5, 1.0, 1.0).set_operation_ratio == 0.5

            # Small positive values for local_connectivity and bandwidth
            @test UMAP.SourceViewParams(1.0, 0.01, 0.01) isa UMAP.SourceViewParams
            @test UMAP.SourceViewParams(1.0, 1e-10, 1e-10) isa UMAP.SourceViewParams
        end

        @testset "Invalid Construction - Should Error" begin
            # set_operation_ratio out of [0, 1]
            @test_throws ArgumentError UMAP.SourceViewParams(-0.1, 1.0, 1.0)
            @test_throws ArgumentError UMAP.SourceViewParams(1.1, 1.0, 1.0)
            @test_throws ArgumentError UMAP.SourceViewParams(2.0, 1.0, 1.0)

            # local_connectivity must be > 0
            @test_throws ArgumentError UMAP.SourceViewParams(1.0, 0.0, 1.0)
            @test_throws ArgumentError UMAP.SourceViewParams(1.0, -1.0, 1.0)

            # bandwidth must be > 0
            @test_throws ArgumentError UMAP.SourceViewParams(1.0, 1.0, 0.0)
            @test_throws ArgumentError UMAP.SourceViewParams(1.0, 1.0, -1.0)
        end
    end

    # -------------------------------------------------------------------------
    # SourceGlobalParams
    # -------------------------------------------------------------------------

    @testset "SourceGlobalParams" begin
        @testset "Valid Construction" begin
            # Basic construction
            params = UMAP.SourceGlobalParams(0.5)
            @test params.mix_ratio == 0.5
            @test params isa UMAP.SourceGlobalParams{Float64}

            # With explicit type
            params_f32 = UMAP.SourceGlobalParams{Float32}(0.3)
            @test params_f32.mix_ratio isa Float32
        end

        @testset "Boundary Values" begin
            @test UMAP.SourceGlobalParams(0.0).mix_ratio == 0.0
            @test UMAP.SourceGlobalParams(1.0).mix_ratio == 1.0
            @test UMAP.SourceGlobalParams(0.5).mix_ratio == 0.5
        end

        @testset "Invalid Construction - Should Error" begin
            # mix_ratio must be in [0, 1]
            @test_throws ArgumentError UMAP.SourceGlobalParams(-0.1)
            @test_throws ArgumentError UMAP.SourceGlobalParams(1.1)
            @test_throws ArgumentError UMAP.SourceGlobalParams(2.0)
        end
    end

    @testset "fuzzy_simplicial_set tests" begin
        knns = [2 3 2; 3 1 1]
        dists = [1.5 .5 .5; 2. 1.5 2.]
        knn_params = DescentNeighbors(2, Euclidean())
        src_params = SourceViewParams(1, 1, 1)
        @testset "fit tests" begin
            @testset "simple test" begin
                umap_graph = @inferred fuzzy_simplicial_set((knns, dists), knn_params, src_params)
                @test issymmetric(umap_graph)
                @test all(0. .<= umap_graph .<= 1.)
                @test size(umap_graph) == (3, 3)
            end
            @testset "named tuple tests" begin
                umap_graph = fuzzy_simplicial_set((knns, dists), knn_params, src_params)
                view_umap_graph = @inferred fuzzy_simplicial_set((view=(knns, dists),), (view=knn_params,), (view=src_params,))
                @test all(umap_graph .≈ view_umap_graph.view)
            end
        end
        @testset "transform tests" begin
            @testset "simple test" begin
                # bogus data just for n_points
                data = [rand(2) for _ in 1:5]
                umap_graph = @inferred fuzzy_simplicial_set(data, (knns, dists), knn_params, src_params)
                @test !issymmetric(umap_graph) # we don't symmetrize transform umap graph
                @test size(umap_graph) == (5, 3)
            end
        end

    end
    @testset "coalesce views tests" begin
        knns_1 = [2 3 2; 3 1 1]
        dists_1 = [1.5 .5 .5; 2. 1.5 2.]
        knn_params = DescentNeighbors(2, Euclidean())
        src_params = SourceViewParams(1, 1, 1)
        knns_2 = [1 2 3; 3 2 1]
        dists_2 = [0.4 0.9 1.2; 0.8 1.1 2.1]
        view_knns = (view_1=(knns_1, dists_1), view_2=(knns_2, dists_2))
        view_knn_params = (view_1=knn_params, view_2=knn_params)
        view_src_params = (view_1=src_params, view_2=src_params)
        view_graphs = fuzzy_simplicial_set(view_knns, view_knn_params, view_src_params)
        gbl_params = SourceGlobalParams(0.5)
        @inferred coalesce_views(view_graphs, gbl_params)
        @inferred coalesce_views(view_graphs.view_1, nothing)
    end


    @testset "smooth_knn_dists" begin
        dists = [0., 1., 2., 3., 4., 5.]
        rho = 1
        k = 6
        local_connectivity = 1
        bandwidth = 1.
        niter = 64
        sigma = smooth_knn_dist(dists, rho, k, bandwidth, niter)
        psum(ds, r, s) = sum(exp.(-max.(ds .- r, 0.) ./ s))
        @test psum(dists, rho, sigma) - log2(k)*bandwidth < SMOOTH_K_TOLERANCE

        knn_dists = [0. 0. 0.;
                    1. 2. 3.;
                    2. 4. 5.;
                    3. 4. 5.;
                    4. 6. 6.;
                    5. 6. 10.]
        src_params = SourceViewParams(1, local_connectivity, bandwidth)
        rhos, sigmas = smooth_knn_dists(knn_dists, k, src_params)
        @test rhos == [1., 2., 3.]
        diffs = [psum(knn_dists[:,i], rhos[i], sigmas[i]) for i in 1:3] .- log2(6)
        @test all(diffs .< SMOOTH_K_TOLERANCE)

        knn_dists = [0. 0. 0.;
                    0. 1. 2.;
                    0. 2. 3.]
        rhos, sigmas = smooth_knn_dists(knn_dists, 2, src_params)
        @test rhos == [0., 1., 2.]
        src_params = SourceViewParams(1, 1.5, 1)
        rhos, sigmas = smooth_knn_dists(knn_dists, 2, src_params)
        @test rhos == [0., 1.5, 2.5]
    end

    @testset "compute_membership_strengths" begin
        knns = [1 2 3; 2 1 2]
        dists = [0. 0. 0.; 2. 2. 3.]
        rhos = [2., 1., 4.]
        sigmas = [1., 1., 1.]
        true_rows = [1, 2, 2, 1, 3, 2]
        true_cols = [1, 1, 2, 2, 3, 3]
        true_vals = [0., 1., 0., exp(-1.), 0., 1.]
        rows, cols, vals = compute_membership_strengths(knns, dists, rhos, sigmas)
        @test rows == true_rows
        @test cols == true_cols
        @test vals == true_vals
    end

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

            S = sprand(100, 100, 0.1)
            @inferred _fuzzy_set_union(S)
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

            S = sprand(100, 100, 0.1)
            @inferred _fuzzy_set_intersection(S)
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

            # Single nonzero value
            single_A = sparse([1, 2], [1, 2], [0.5, 0.3], 3, 3)
            single_B = sparse([1, 3], [1, 3], [0.4, 0.6], 3, 3)
            res = general_simplicial_set_union(single_A, single_B)
            @test res isa SparseMatrixCSC
        end
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
        @testset "Sparsity Pattern" begin
            S = sprand(100, 100, 0.1)
            T = sprand(100, 100, 0.1)
            S_inds = findall(!iszero, S)
            T_inds = findall(!iszero, T)
            res_union = general_simplicial_set_union(S, T)
            res_inter = general_simplicial_set_intersection(S, T, SourceGlobalParams(0.5))
            union_inds = findall(!iszero, res_union)
            inter_inds = findall(!iszero, res_inter)
            # test that the union of the indices equals the indices of the operations
            S_T_inds = union(Set(S_inds), Set(T_inds))
            @test S_T_inds == Set(union_inds)
            @test S_T_inds == Set(inter_inds)
        end
    end

    # -------------------------------------------------------------------------
    # Local Connectivity and Normalization
    # -------------------------------------------------------------------------

    @testset "reset_local_connectivity tests" begin
        @testset "Basic Properties" begin
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
            @inferred reset_local_connectivity(A, true)
            @inferred reset_local_connectivity(A, false)
            result = reset_local_connectivity(A, false)
            @test result isa SparseMatrixCSC
            @test result ≈ result'  # Still symmetric
        end

        @testset "_norm_sparse tests" begin
            @testset "Column-wise normalization" begin
                A = rand(4, 8) .+ 1e-8  # add 1e-8 to eliminate any possible
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
                A = [0.8 0.0; 0.4 0.0]
                spA = sparse(A)
                result = _norm_sparse(spA)
                @test result isa SparseMatrixCSC
                # First column normalized to max 0.8
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

                A = sprand(5, 20, 0.2)
                A_res = _norm_sparse(A)
                I, J, _ = findnz(A)
                I2, J2, __ = findnz(A_res)
                @test I == I2
                @test J == J2
            end
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
        end

        @testset "_mix_values tests" begin
            @testset "Equal weighting (ratio = 0.5)" begin
                # ratio = 0.5 should give product
                @test _mix_values(4.0, 4.0, 0.5) ≈ 16.0
                @test _mix_values(1.0, 4.0, 0.5) ≈ 4.0
            end

            @testset "Left weighting (ratio < 0.5)" begin
                # ratio = 0 should return x
                @test _mix_values(0.3, 0.7, 0.0) ≈ 0.3
                # Small ratio favors x
                @test _mix_values(2.0, 8.0, 0.1) < 4.0  # closer to 2 than 8
            end

            @testset "Right weighting (ratio > 0.5)" begin
                # ratio = 1 should return y
                @test _mix_values(0.4, 0.8, 1.0) ≈ 0.8
                # Large ratio favors y
                @test _mix_values(2.0, 8.0, 0.9) > 5.0  # closer to 8 than 2
            end

            @testset "Boundary and edge cases" begin
                # Test with 1.0
                @test _mix_values(1.0, 1.0, 0.5) ≈ 1.0

                # Test symmetry
                @test _mix_values(2.0, 8.0, 0.49) ≈ _mix_values(8.0, 2.0, 0.51)
            end
        end
    end

end
