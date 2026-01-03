
@testset "optimize_embedding tests" begin

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
    end

    backend = DI.AutoZygote()
    @testset "OptimizationParams tests" begin
        @test_throws ArgumentError UMAP.OptimizationParams(0, .1, .1, 1)
        @test_throws ArgumentError UMAP.OptimizationParams(1, -0.1, 1., 1)
        @test_throws ArgumentError UMAP.OptimizationParams(1, 1., -0.1, 1)
        @test_throws ArgumentError UMAP.OptimizationParams(1, 1., 1., -1)

        params = UMAP.OptimizationParams(1, 1., 1., 1)
        new_params = UMAP.set_lr(params, 0.5)
        @test new_params.lr == 0.5
        @test params.lr == 1.
    end
    a, b = 1., 1.
    # set a, b to 1. for easy testing
    tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold(3), 
                                   Distances.SqEuclidean(), 
                                   UMAP.UniformInitialization(), 
                                   UMAP.MembershipFnParams(1., 2., a, b))
    opt_params = UMAP.OptimizationParams(10, 0.1, 1., 1)

    v = [1., 1, 1]
    w = [2., 2, 2]
    # test that the gradient calculation is correct - check that the 
    # updated v, w match after applying update
    log_phi(x, y) = log(UMAP._ϕ(x, y, Distances.sqeuclidean, a, b))
    log_1_phi(x, y) = log(1 - UMAP._ϕ(x, y, Distances.sqeuclidean, a, b))
    @testset "optimize_embedding_pos! sqeuclidean" begin
        v_pos_grad = DI.gradient(x_ -> log_phi(x_, w), backend, v)
        w_pos_grad = DI.gradient(x_ -> log_phi(v, x_), backend, w)

        v_updated_true = copy(v) + opt_params.lr * v_pos_grad
        w_updated_true = copy(w) + opt_params.lr * w_pos_grad

        v_copy = copy(v)
        w_copy = copy(w)
        # test the positive updates
        UMAP.update_embedding_pos!(v_copy, w_copy, tgt_params, opt_params, true)
        @test v_copy ≈ v_updated_true
        @test w_copy ≈ w_updated_true
        # test that the reference embedding doesn't change with flag
        v_copy = copy(v)
        w_copy = copy(w)
        UMAP.update_embedding_pos!(v_copy, w_copy, tgt_params, opt_params, false)
        @test v_copy ≈ v_updated_true
        @test w == w_copy
    end

    @testset "optimize_embedding_neg! sqeuclidean" begin
        v_neg_grad = DI.gradient(x_ -> log_1_phi(x_, w), backend, v)
        v_updated_true = copy(v) + opt_params.repulsion_strength * opt_params.lr * v_neg_grad

        v_copy = copy(v)
        w_copy = copy(w)
        # test that negative updates, but unchanging reference
        UMAP.update_embedding_neg!(v_copy, w_copy, tgt_params, opt_params)
        @test v_copy ≈ v_updated_true
        @test w_copy == w
    end
end