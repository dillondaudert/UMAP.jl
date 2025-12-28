using Distances: Euclidean, SqEuclidean, Cityblock, Chebyshev

@testset "Configuration Tests" begin

    # -------------------------------------------------------------------------
    # MembershipFnParams
    # -------------------------------------------------------------------------

    @testset "MembershipFnParams" begin
        @testset "Valid Construction - Explicit a, b" begin
            # Direct construction with all parameters
            params = UMAP.MembershipFnParams(0.1, 1.0, 1.5, 0.8)
            @test params.min_dist == 0.1
            @test params.spread == 1.0
            @test params.a == 1.5
            @test params.b == 0.8
            @test params isa UMAP.MembershipFnParams{Float64}

            # Type promotion
            params_mixed = UMAP.MembershipFnParams(0.1, 1, 1.5, 1)
            @test all(isa.([params_mixed.min_dist, params_mixed.spread,
                           params_mixed.a, params_mixed.b], Float64))
        end

        @testset "Valid Construction - Auto-fit a, b" begin
            # Fit a, b from min_dist and spread
            params = UMAP.MembershipFnParams(0.1, 1.0)
            @test params.min_dist == 0.1
            @test params.spread == 1.0
            @test params.a > 0  # Should be fitted to reasonable values
            @test params.b > 0

            # Test that different min_dist/spread give different a, b
            params1 = UMAP.MembershipFnParams(0.1, 1.0)
            params2 = UMAP.MembershipFnParams(0.5, 2.0)
            @test params1.a != params2.a || params1.b != params2.b

            # Test with nothing arguments (should trigger fit)
            params_nothing = UMAP.MembershipFnParams(0.1, 1.0, nothing, nothing)
            @test params_nothing.a > 0
            @test params_nothing.b > 0
        end

        @testset "Mutability" begin
            # MembershipFnParams is mutable - test that we can modify it
            params = UMAP.MembershipFnParams(0.1, 1.0, 1.0, 1.0)
            params.a = 2.0
            @test params.a == 2.0

            # Note: This is important for potential adaptive algorithms
        end

        @testset "Invalid Construction - Should Error" begin
            # min_dist must be > 0
            @test_throws ArgumentError UMAP.MembershipFnParams(0.0, 1.0, 1.0, 1.0)
            @test_throws ArgumentError UMAP.MembershipFnParams(-0.1, 1.0, 1.0, 1.0)

            # spread must be > 0
            @test_throws ArgumentError UMAP.MembershipFnParams(0.1, 0.0, 1.0, 1.0)
            @test_throws ArgumentError UMAP.MembershipFnParams(0.1, -1.0, 1.0, 1.0)

            # Auto-fit should also enforce constraints
            @test_throws ArgumentError UMAP.MembershipFnParams(0.0, 1.0)
            @test_throws ArgumentError UMAP.MembershipFnParams(0.1, 0.0)
        end

        @testset "Boundary Values" begin
            # Very small but valid values
            params_small = UMAP.MembershipFnParams(1e-6, 1e-6, 1.0, 1.0)
            @test params_small isa UMAP.MembershipFnParams

            # Large values
            params_large = UMAP.MembershipFnParams(10.0, 100.0, 5.0, 5.0)
            @test params_large isa UMAP.MembershipFnParams
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test that fit_ab produces reasonable a, b values for typical inputs
        # TODO: Test relationship between (min_dist, spread) and resulting (a, b)
        # TODO: Validate that membership function with fitted params matches expected curve
        # TODO: Test edge cases for fit_ab (very small/large min_dist or spread)
    end

    # -------------------------------------------------------------------------
    # TargetParams
    # -------------------------------------------------------------------------

    @testset "TargetParams" begin
        @testset "Valid Construction" begin
            # Standard Euclidean manifold with spectral init
            memb_params = UMAP.MembershipFnParams(0.1, 1.0)
            params = UMAP.TargetParams(
                UMAP._EuclideanManifold(2),
                SqEuclidean(),
                UMAP.SpectralInitialization(),
                memb_params
            )
            @test params.manifold == UMAP._EuclideanManifold(2)
            @test params.metric == SqEuclidean()
            @test params.init isa UMAP.SpectralInitialization
            @test params.memb_params === memb_params

            # With UniformInitialization
            params_uniform = UMAP.TargetParams(
                UMAP._EuclideanManifold(3),
                Euclidean(),
                UMAP.UniformInitialization(),
                memb_params
            )
            @test params_uniform.init isa UMAP.UniformInitialization

            # Different dimensions
            @test UMAP.TargetParams(
                UMAP._EuclideanManifold(10),
                SqEuclidean(),
                UMAP.SpectralInitialization(),
                memb_params
            ).manifold == UMAP._EuclideanManifold(10)
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test type stability with different metric types
        # TODO: Test that manifold dimension matches expected embedding dimension
        # TODO: Add tests when other manifolds are supported
        # TODO: Test custom initialization strategies
    end

    # -------------------------------------------------------------------------
    # OptimizationParams
    # -------------------------------------------------------------------------

    @testset "OptimizationParams" begin
        @testset "Valid Construction" begin
            params = UMAP.OptimizationParams(100, 1.0, 1.0, 5)
            @test params.n_epochs == 100
            @test params.lr == 1.0
            @test params.repulsion_strength == 1.0
            @test params.neg_sample_rate == 5
        end

        @testset "Boundary Values" begin
            # Minimum valid values
            @test UMAP.OptimizationParams(1, 0.0, 0.0, 0).n_epochs == 1

            # Large values
            params_large = UMAP.OptimizationParams(10000, 10.0, 5.0, 100)
            @test params_large.n_epochs == 10000
        end

        @testset "Invalid Construction - Should Error" begin
            # n_epochs must be > 0
            @test_throws ArgumentError UMAP.OptimizationParams(0, 1.0, 1.0, 5)
            @test_throws ArgumentError UMAP.OptimizationParams(-1, 1.0, 1.0, 5)

            # lr must be >= 0
            @test_throws ArgumentError UMAP.OptimizationParams(100, -0.1, 1.0, 5)

            # neg_sample_rate must be >= 0
            @test_throws ArgumentError UMAP.OptimizationParams(100, 1.0, 1.0, -1)
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test typical parameter combinations (e.g., n_epochs=300, lr=1.0)
        # TODO: Test effect of repulsion_strength (no constraint currently, should there be?)
        # TODO: Test learning rate decay behavior (tested in optimize.jl)
        # TODO: Document recommended ranges for each parameter
    end

    # -------------------------------------------------------------------------
    # UMAPConfig - Integration Tests
    # -------------------------------------------------------------------------

    @testset "UMAPConfig" begin
        @testset "Valid Construction" begin
            # Create all sub-configs
            knn_params = UMAP.DescentNeighbors(15, Euclidean())
            src_params = UMAP.SourceViewParams(1.0, 1.0, 1.0)
            gbl_params = UMAP.SourceGlobalParams(0.5)
            memb_params = UMAP.MembershipFnParams(0.1, 1.0)
            tgt_params = UMAP.TargetParams(
                UMAP._EuclideanManifold(2),
                SqEuclidean(),
                UMAP.SpectralInitialization(),
                memb_params
            )
            opt_params = UMAP.OptimizationParams(100, 1.0, 1.0, 5)

            config = UMAP.UMAPConfig(knn_params, src_params, gbl_params, tgt_params, opt_params)
            @test config.knn_params === knn_params
            @test config.src_params === src_params
            @test config.gbl_params === gbl_params
            @test config.tgt_params === tgt_params
            @test config.opt_params === opt_params
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test type stability of UMAPConfig
        # TODO: Test with different parameter combinations
        # TODO: Test that config can be modified via Setfield.jl
        # TODO: Create factory functions for common configurations
    end

    # -------------------------------------------------------------------------
    # Result Types
    # -------------------------------------------------------------------------

    @testset "Result Types" begin
        @testset "UMAPResult Structure" begin
            # Test that we can construct result (actual construction tested in fit tests)
            # This is more of a smoke test for the struct definition

            # Note: Actual UMAPResult construction is tested in integration tests
            # Here we just verify the type exists and has expected fields
            @test fieldnames(UMAP.UMAPResult) == (:data, :embedding, :config, :knns_dists, :fs_sets, :graph)
        end

        @testset "UMAPTransformResult Structure" begin
            @test fieldnames(UMAP.UMAPTransformResult) == (:data, :embedding, :knns_dists, :fs_sets, :graph)
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test result accessors/utility functions if added
        # TODO: Test serialization/deserialization of results
        # TODO: Test that results contain expected types for each field
    end

    # =========================================================================
    # ADDITIONAL TESTING SUGGESTIONS
    # =========================================================================
    #
    # 1. INTEGRATION TESTS:
    #    - Test that default parameters from fit() create valid configs
    #    - Test parameter propagation through the full pipeline
    #    - Test that configs can be reused for multiple fits
    #
    # 2. TYPE STABILITY TESTS:
    #    - Use @inferred to check type stability of constructors
    #    - Verify no type instabilities in config creation
    #
    # 3. SERIALIZATION TESTS:
    #    - Test saving/loading configurations
    #    - Test JSON/BSON serialization if needed
    #
    # 4. DOCUMENTATION TESTS:
    #    - Verify all parameter constraints are documented
    #    - Test example configurations from documentation
    #
    # 5. PERFORMANCE TESTS:
    #    - Benchmark config creation overhead
    #    - Test that configs don't allocate unnecessarily
    #
    # 6. MULTI-VIEW CONFIG TESTS:
    #    - Test NamedTuple configurations for multiple views
    #    - Verify consistent keys across view parameters
    #
    # =========================================================================

end
