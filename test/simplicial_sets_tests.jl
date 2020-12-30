@testset "simplicial sets tests" begin
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
end
