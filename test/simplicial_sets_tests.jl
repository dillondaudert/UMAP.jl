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

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Verify that local connectivity constraint is satisfied
        # TODO: Test with matrices of different sparsity patterns
        # TODO: Compare results with/without metric reset

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
end
