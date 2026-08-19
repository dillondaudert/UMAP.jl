# tests for ExactNeighbors and the NearestNeighbors.jl package extension
import NearestNeighbors

@testset "exact neighbors tests" begin

    @testset "ExactNeighbors construction" begin
        params = UMAP.ExactNeighbors(10, Distances.Euclidean())
        @test params isa UMAP.NeighborParams
        @test params.n_neighbors == 10
        @test params.metric == Distances.Euclidean()
        @test params.kwargs == NamedTuple()
        # defaults
        @test UMAP.ExactNeighbors(5).metric == Distances.Euclidean()
        # tree constructor kwargs
        @test UMAP.ExactNeighbors(5, Distances.Euclidean(), (leafsize=5,)).kwargs == (leafsize=5,)
        # invariants
        @test_throws ArgumentError UMAP.ExactNeighbors(0, Distances.Euclidean())
        @test_throws ArgumentError UMAP.ExactNeighbors(-3, Distances.Euclidean())
    end

    @testset "knn_search agrees with brute force (fit)" begin
        data = rand(6, 40)
        # KDTree-compatible metrics
        for metric in (Distances.Euclidean(), Distances.Cityblock())
            knns, dists = UMAP.knn_search(data, UMAP.ExactNeighbors(5, metric))
            @test size(knns) == (5, 40)
            @test size(dists) == (5, 40)

            dist_mat = [metric(view(data, :, i), view(data, :, j)) for i in 1:40, j in 1:40]
            knns_bf, dists_bf = UMAP._knn_from_dists(dist_mat, 5)
            @test knns == knns_bf
            @test dists ≈ dists_bf
        end

        # BallTree path (Hamming has no KDTree support); ties are common with
        # this metric so compare only the distances
        bdata = float.(rand(0:1, 8, 30))
        knns, dists = UMAP.knn_search(bdata, UMAP.ExactNeighbors(4, Distances.Hamming()))
        dist_mat = [Distances.Hamming()(view(bdata, :, i), view(bdata, :, j)) for i in 1:30, j in 1:30]
        _, dists_bf = UMAP._knn_from_dists(dist_mat, 4)
        @test dists ≈ dists_bf
    end

    @testset "knn_search agrees with brute force (transform)" begin
        data = rand(6, 40)
        queries = rand(6, 9)
        knns, dists = UMAP.knn_search(data, queries, UMAP.ExactNeighbors(5, Distances.Euclidean()), nothing)
        @test size(knns) == (5, 9)
        @test size(dists) == (5, 9)

        dist_mat = [Distances.euclidean(view(data, :, i), view(queries, :, j)) for i in 1:40, j in 1:9]
        knns_bf, dists_bf = UMAP._knn_from_dists(dist_mat, 5; ignore_diagonal=false)
        @test knns == knns_bf
        @test dists ≈ dists_bf
    end

    @testset "vector of vectors input" begin
        vdata = [rand(6) for _ in 1:30]
        mdata = reduce(hcat, vdata)
        knns_v, dists_v = UMAP.knn_search(vdata, UMAP.ExactNeighbors(4, Distances.Euclidean()))
        knns_m, dists_m = UMAP.knn_search(mdata, UMAP.ExactNeighbors(4, Distances.Euclidean()))
        @test knns_v == knns_m
        @test dists_v ≈ dists_m
    end

    @testset "NamedTuple views can mix knn backends" begin
        data = (view_1 = rand(4, 30), view_2 = rand(6, 30))
        knn_params = (view_1 = UMAP.ExactNeighbors(4, Distances.Euclidean()),
                      view_2 = UMAP.DescentNeighbors(4, Distances.Euclidean()))
        knns_dists = UMAP.knn_search(data, knn_params)
        @test keys(knns_dists) == (:view_1, :view_2)
        @test size(knns_dists.view_1[1]) == (4, 30)
        @test size(knns_dists.view_2[1]) == (4, 30)
    end

    @testset "fit and transform end-to-end" begin
        Random.seed!(1)
        data = hcat(randn(5, 40) .- 3, randn(5, 40) .+ 3)
        knn_params = UMAP.ExactNeighbors(8, Distances.Euclidean())
        src_params = UMAP.SourceViewParams(1., 1., 1.)
        gbl_params = UMAP.SourceGlobalParams(0.5)
        tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold(2), Distances.SqEuclidean(),
                                       UMAP.SpectralInitialization(), UMAP.MembershipFnParams(0.1, 1.))
        opt_params = UMAP.OptimizationParams(100, 1., 1., 5)

        result = UMAP.fit(data, knn_params, src_params, gbl_params, tgt_params, opt_params)
        @test result isa UMAP.UMAPResult
        @test size(result.embedding) == (2, 80)
        @test all(isfinite, result.embedding)
        @test result.config.knn_params === knn_params

        # 2-arg transform re-uses the ExactNeighbors params from the fit config
        queries = randn(5, 10) .- 3
        tresult = UMAP.transform(result, queries)
        @test tresult isa UMAP.UMAPTransformResult
        @test size(tresult.embedding) == (2, 10)
        @test all(isfinite, tresult.embedding)
    end

end
