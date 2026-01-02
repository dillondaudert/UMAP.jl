import Accessors: @set

@testset "Configuration Tests" begin

    @testset "Create view config tests" begin
        X = rand(10, 20)
        nn = 4
        metric = Euclidean()
        set_op_ratio = 0.5
        local_conn = 1
        bandwidth = 1.
        knn_kwargs = NamedTuple()
        desc_params = UMAP.DescentNeighbors(nn, metric, knn_kwargs)
        prec_params = UMAP.PrecomputedNeighbors(nn, X)
        src_params = UMAP.SourceViewParams(set_op_ratio, local_conn, bandwidth)

        @testset "create_view_config FIT tests" begin
            view_params = UMAP.create_view_config(;
                data_or_dists=X,
                n_neighbors=nn,
                metric=metric,
                set_operation_ratio=set_op_ratio,
                local_connectivity=local_conn,
                bandwidth=bandwidth,
                knn_kwargs=knn_kwargs
            )
            @test view_params == (desc_params, src_params)
            @inferred UMAP.create_view_config(;
                data_or_dists=X,
                n_neighbors=nn,
                metric=metric,
                set_operation_ratio=set_op_ratio,
                local_connectivity=local_conn,
                bandwidth=bandwidth,
                knn_kwargs=knn_kwargs
            )
            view_pre_params = UMAP.create_view_config(;
                data_or_dists=X,
                n_neighbors=nn,
                metric=:precomputed,
                set_operation_ratio=set_op_ratio,
                local_connectivity=local_conn,
                bandwidth=bandwidth,
                knn_kwargs=knn_kwargs
            )
            @test view_pre_params == (prec_params, src_params)
        end
        @testset "create_view_config TRANSFORM tests" begin
            view_params = UMAP.create_view_config(prec_params, src_params; data_or_dists=X)
            @test view_params == (prec_params, src_params)

            # test overwrite precomputed params n_neighbors
            view_params = UMAP.create_view_config(prec_params, src_params; data_or_dists=X, n_neighbors=12)
            @test view_params == ((@set prec_params.n_neighbors = 12), src_params)
            # create parameters from existing structs
            @inferred UMAP.create_view_config(desc_params, src_params; data_or_dists=X)
            view_params = UMAP.create_view_config(desc_params, src_params; data_or_dists=X)
            @test view_params == (desc_params, src_params)

            # overwrite by kw arg various fields 
            view_params = UMAP.create_view_config(desc_params, src_params; data_or_dists=X, n_neighbors=10)
            @test view_params == ((@set desc_params.n_neighbors = 10), src_params)

            # overwrite metric 
            view_params = UMAP.create_view_config(desc_params, src_params; data_or_dists=X, metric=SqEuclidean())
            @test view_params == ((@set desc_params.metric = SqEuclidean()), src_params)

            # ovewrite kwargs
            view_params = UMAP.create_view_config(desc_params, src_params; data_or_dists=X, knn_kwargs=(knn=20,))
            @test view_params == ((@set desc_params.kwargs = (knn=20,)), src_params)

            #overwrite src params
            new_src_params = UMAP.SourceViewParams(0.8, 2, 2.)
            view_params = UMAP.create_view_config(desc_params, src_params; data_or_dists=X,
                                                  set_operation_ratio=0.8,
                                                  local_connectivity=2,
                                                  bandwidth=2.)
            @test view_params == (desc_params, new_src_params)

        end

    end

    @testset "create_config tests" begin
        X = rand(5, 20)
        Y = rand(3, 20)
        view_1_args = (
            data_or_dists = X,
            n_neighbors = 10,
            metric = Distances.Euclidean(),
            knn_kwargs = NamedTuple(),
            set_operation_ratio = 0.8,
            local_connectivity = 1,
            bandwidth = 1.
        )
        view_1_params = (
            UMAP.DescentNeighbors(10, Distances.Euclidean()),
            UMAP.SourceViewParams(0.8, 1, 1.)
        )
        view_2_args = (
            data_or_dists = Y,
            n_neighbors = 8,
            metric = :precomputed,
            knn_kwargs = NamedTuple(),
            set_operation_ratio = 0.7,
            local_connectivity = 2,
            bandwidth = 1.2
        )
        view_2_params = (
            UMAP.PrecomputedNeighbors(8, Y),
            UMAP.SourceViewParams(0.7, 2, 1.2)
        )
        config_args = (
            global_mix_ratio = 0.5,
            min_dist = 0.1,
            spread = 1.,
            n_components = 5,
            target_metric = Distances.SqEuclidean(),
            init = UMAP.UniformInitialization(),
            n_epochs = 10,
            learning_rate = 1.,
            repulsion_strength = 0.1,
            neg_sample_rate = 5,
        )
        gbl_params = UMAP.SourceGlobalParams(0.5)
        memb_params = UMAP.MembershipFnParams(0.1, 1.)
        tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold(5), Distances.SqEuclidean(), UMAP.UniformInitialization(), memb_params)
        opt_params = UMAP.OptimizationParams(10, 1., 0.1, 5)

        # test single view, doesn't create named tuple of view params
        single_view_res = UMAP.create_config(view_1_args; pairs(config_args)...)
        @test single_view_res[1] == X
        @test single_view_res[2:3] == view_1_params
        @test single_view_res[4] == gbl_params
        @test typeof(single_view_res[5]) == typeof(tgt_params)
        @test single_view_res[5].manifold == tgt_params.manifold
        @test single_view_res[6] == opt_params

        # test multiple views
        multi_view_res = UMAP.create_config([view_1_args, view_2_args]; pairs(config_args)...)
        NT = NamedTuple{(:view_1, :view_2)}
        @test multi_view_res[1] isa NT
        @test multi_view_res[2] isa NT
        @test multi_view_res[3] isa NT
        @test multi_view_res[1] == (view_1=X, view_2=Y)
        @test multi_view_res[2] == (view_1=view_1_params[1], view_2=view_2_params[1])
        @test multi_view_res[3] == (view_1=view_1_params[2], view_2=view_2_params[2])
        
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
