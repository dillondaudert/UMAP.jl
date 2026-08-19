# end-to-end integration tests: fit/transform on clustered data with quality checks

# per-label centroids of an embedding, indexed by label
function _centroids(embedding, labels)
    return [vec(sum(embedding[:, labels .== l], dims=2)) ./ count(==(l), labels)
            for l in sort(unique(labels))]
end

# fraction of points whose nearest centroid is their own label's centroid
function _label_purity(embedding, labels, centroids)
    n_correct = count(1:size(embedding, 2)) do i
        argmin([norm(embedding[:, i] .- c) for c in centroids]) == labels[i]
    end
    return n_correct / size(embedding, 2)
end

@testset "integration tests" begin

    Random.seed!(1234)
    # three well-separated gaussian clusters in R^10
    n_per_cluster = 60
    centers = [zeros(10), vcat(10., zeros(9)), vcat(0., 10., zeros(8))]
    data = reduce(hcat, [c .+ randn(10) for c in centers for _ in 1:n_per_cluster])
    labels = repeat(1:3, inner=n_per_cluster)
    n_points = length(labels)

    queries = reduce(hcat, [c .+ randn(10) for c in centers for _ in 1:10])
    query_labels = repeat(1:3, inner=10)

    result = UMAP.fit(data)
    embed_centroids = _centroids(result.embedding, labels)

    @testset "fit recovers cluster structure" begin
        @test size(result.embedding) == (2, n_points)
        @test all(isfinite, result.embedding)
        @test _label_purity(result.embedding, labels, embed_centroids) > 0.9
        @test UMAP.trustworthiness(data, result.embedding, 10, Distances.euclidean) > 0.8
    end

    @testset "transform places new points in the right clusters" begin
        tresult = UMAP.transform(result, queries)
        @test size(tresult.embedding) == (2, length(query_labels))
        @test all(isfinite, tresult.embedding)
        @test _label_purity(tresult.embedding, query_labels, embed_centroids) > 0.9
    end

    @testset "multi-view fit" begin
        # two views of the same points; each view sees both cluster-separating dims
        mv_data = (view_1 = data[1:5, :], view_2 = data[[1, 2, 6, 7, 8], :])
        mv_knn_params = (view_1 = UMAP.DescentNeighbors(10, Distances.Euclidean()),
                         view_2 = UMAP.DescentNeighbors(10, Distances.Euclidean()))
        mv_src_params = (view_1 = UMAP.SourceViewParams(1., 1., 1.),
                         view_2 = UMAP.SourceViewParams(1., 1., 1.))
        gbl_params = UMAP.SourceGlobalParams(0.5)
        tgt_params = UMAP.TargetParams(UMAP._EuclideanManifold(2), Distances.SqEuclidean(),
                                       UMAP.SpectralInitialization(), UMAP.MembershipFnParams(0.1, 1.))
        opt_params = UMAP.OptimizationParams(200, 1., 1., 5)

        mv_result = UMAP.fit(mv_data, mv_knn_params, mv_src_params, gbl_params, tgt_params, opt_params)
        # views are coalesced into a single umap graph and embedding
        @test mv_result.graph isa SparseMatrixCSC
        @test size(mv_result.embedding) == (2, n_points)
        @test all(isfinite, mv_result.embedding)
        mv_centroids = _centroids(mv_result.embedding, labels)
        @test _label_purity(mv_result.embedding, labels, mv_centroids) > 0.9

        # multi-view transform with matching NamedTuple queries
        mv_queries = (view_1 = queries[1:5, :], view_2 = queries[[1, 2, 6, 7, 8], :])
        mv_tresult = UMAP.transform(mv_result, mv_queries)
        @test size(mv_tresult.embedding) == (2, length(query_labels))
        @test _label_purity(mv_tresult.embedding, query_labels, mv_centroids) > 0.8
    end

    @testset "precomputed distances fit and transform" begin
        pdists = [Distances.euclidean(view(data, :, i), view(data, :, j))
                  for i in 1:n_points, j in 1:n_points]
        presult = UMAP.fit(pdists; metric=:precomputed)
        @test size(presult.embedding) == (2, n_points)
        @test all(isfinite, presult.embedding)
        p_centroids = _centroids(presult.embedding, labels)
        @test _label_purity(presult.embedding, labels, p_centroids) > 0.9

        # transform new points via their distances to the original data
        qdists = [Distances.euclidean(view(data, :, i), view(queries, :, j))
                  for i in 1:n_points, j in 1:length(query_labels)]
        knn_params = UMAP.PrecomputedNeighbors(15, qdists)
        cfg = presult.config
        opt_params = UMAP.OptimizationParams(30, 0.25, 1., 5)
        ptresult = UMAP.transform(presult, qdists, knn_params, cfg.src_params,
                                  cfg.gbl_params, cfg.tgt_params, opt_params)
        @test size(ptresult.embedding) == (2, length(query_labels))
        @test _label_purity(ptresult.embedding, query_labels, p_centroids) > 0.9
    end

end
