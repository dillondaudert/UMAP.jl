using Distances: Euclidean, SqEuclidean, Cityblock, Chebyshev

@testset "Configuration Tests" begin

    # =============================================================================
    # TEST PLAN FOR config.jl
    # =============================================================================
    #
    # This test suite validates the configuration types and their constraints.
    # Each type has inner constructors that enforce invariants - these must be tested
    # to ensure invalid configurations are rejected and valid ones are accepted.
    #
    # COVERAGE GOALS:
    # 1. Test all constructor constraints (boundary conditions, invalid values)
    # 2. Test type promotion and conversion
    # 3. Test default constructors and keyword argument variants
    # 4. Test that valid configurations are accepted
    # 5. Test parameter propagation through UMAPConfig
    #
    # TESTING STRATEGY:
    # - For each type, test both valid and invalid parameter values
    # - Test boundary conditions (0, negative, edge cases)
    # - Test type stability and promotion
    # - Test that constraints are enforced (should error on invalid input)
    #
    # =============================================================================

    # -------------------------------------------------------------------------
    # NeighborParams Types
    # -------------------------------------------------------------------------

    @testset "DescentNeighbors" begin
        # VALID CASES
        @testset "Valid Construction" begin
            # Basic construction with metric
            params = UMAP.DescentNeighbors(15, Euclidean())
            @test params.n_neighbors == 15
            @test params.metric == Euclidean()
            @test params.kwargs == NamedTuple()

            # Construction with kwargs
            params_kwargs = UMAP.DescentNeighbors(20, SqEuclidean(), (max_iters=10,))
            @test params_kwargs.n_neighbors == 20
            @test params_kwargs.kwargs.max_iters == 10

            # Test with different metrics
            @test UMAP.DescentNeighbors(10, Cityblock()).metric == Cityblock()
            @test UMAP.DescentNeighbors(10, Chebyshev()).metric == Chebyshev()
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Add tests for invalid n_neighbors (0, negative) if validation added
        # TODO: Test with custom distance functions
        # TODO: Test kwargs propagation to nndescent
    end

    @testset "PrecomputedNeighbors" begin
        # VALID CASES
        @testset "Valid Construction" begin
            # With distance matrix
            dists = rand(100, 100)
            params = UMAP.PrecomputedNeighbors(15, dists)
            @test params.n_neighbors == 15
            @test params.dists_or_graph === dists

            # With KNN graph (tested in integration with neighbors.jl)
            # This is tested in neighbors_tests.jl since it requires NND types
        end

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test that n_neighbors doesn't exceed available neighbors in graph/matrix
        # TODO: Test with empty/malformed distance matrices
        # TODO: Test with non-square distance matrices (for transform case)
    end

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

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test typical values used in practice (local_connectivity=1, bandwidth=1)
        # TODO: Test extreme but valid values (very small bandwidth, large local_connectivity)
        # TODO: Document recommended parameter ranges based on data characteristics
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

        # SUGGESTIONS FOR ADDITIONAL TESTS:
        # TODO: Test semantic meaning of mix_ratio values (0 vs 0.5 vs 1)
        # TODO: Test with multiple views to validate mix_ratio behavior
    end

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
