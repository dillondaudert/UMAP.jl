
@testset "Nearest Neighbors Tests" begin
    import UMAP: DescentNeighbors, PrecomputedNeighbors, knn_search, _knn_from_dists
    import NearestNeighborDescent as NND

    # -------------------------------------------------------------------------
    # NeighborParams Types
    # -------------------------------------------------------------------------

    @testset "DescentNeighbors" begin
        # VALID CASES
        @testset "Valid Construction" begin
            # Basic construction with metric
            params = UMAP.DescentNeighbors(15, Distances.Euclidean())
            @test params.n_neighbors == 15
            @test params.metric == Distances.Euclidean()
            @test params.kwargs == NamedTuple()

            # Construction with kwargs
            params_kwargs = UMAP.DescentNeighbors(20, Distances.SqEuclidean(), (max_iters=10,))
            @test params_kwargs.n_neighbors == 20
            @test params_kwargs.kwargs.max_iters == 10

            # Test with different metrics
            @test UMAP.DescentNeighbors(10, Distances.Cityblock()).metric == Distances.Cityblock()
            @test UMAP.DescentNeighbors(10, Distances.Chebyshev()).metric == Distances.Chebyshev()

            @test_throws ErrorException UMAP.DescentNeighbors(0, Distances.Euclidean())
        end

    end

    @testset "PrecomputedNeighbors" begin
        # VALID CASES
        @testset "Valid Construction" begin
            # With distance matrix
            dists = rand(100, 100)
            params = UMAP.PrecomputedNeighbors(15, dists)
            @test params.n_neighbors == 15
            @test params.dists_or_graph === dists

            @test_throws ErrorException UMAP.PrecomputedNeighbors(0, dists)

            # tests with NND / KNNGraph below
        end
    end

    # -------------------------------------------------------------------------
    # FIT: Single View Tests
    # -------------------------------------------------------------------------

    @testset "FIT: Single View" begin

        @testset "DescentNeighbors - Basic Functionality" begin
            data = [rand(10) for _ in 1:100]
            knn_params = UMAP.DescentNeighbors(5, Distances.Euclidean())
            knn_params_kw = UMAP.DescentNeighbors(5, Distances.Euclidean(), (max_iters=5,))
            knn_params_bad_kw = UMAP.DescentNeighbors(5, Distances.Euclidean(), (bad_kw=10.,))
            RT = Tuple{Array{Int,2},Array{Float64,2}}

            # Type stability
            @inferred knn_search(data, knn_params)

            # Basic return type
            @test knn_search(data, knn_params) isa RT

            # Kwargs propagation
            @test knn_search(data, knn_params_kw) isa RT
            @test_throws MethodError knn_search(data, knn_params_bad_kw)

            # Named tuple single view
            @inferred knn_search((view=data,), (view=knn_params,))
            res = knn_search((view=data,), (view=knn_params,))
            @test res.view isa RT
        end

        @testset "DescentNeighbors - Output Shape and Properties" begin
            n_points, n_neighbors = 50, 10
            data = [rand(5) for _ in 1:n_points]
            knn_params = DescentNeighbors(n_neighbors, Distances.Euclidean())

            knns, dists = knn_search(data, knn_params)

            # Shape tests
            @test size(knns) == (n_neighbors, n_points)
            @test size(dists) == (n_neighbors, n_points)

            # Index validity (all indices should be in 1:n_points)
            @test all(1 .<= knns .<= n_points)

            # Distance properties
            @test all(dists .>= 0)  # Distances should be non-negative
            @test eltype(dists) == Float64

            # Each column should have unique neighbors (no duplicates)
            for col in eachcol(knns)
                @test length(unique(col)) == n_neighbors
            end
        end

        @testset "DescentNeighbors - Different Metrics" begin
            data = [rand(8) for _ in 1:30]
            RT = Tuple{Array{Int,2},Array{Float64,2}}

            # Test various metrics
            @test knn_search(data, DescentNeighbors(5, Distances.Euclidean())) isa RT
            @test knn_search(data, DescentNeighbors(5, Distances.SqEuclidean())) isa RT
            @test knn_search(data, DescentNeighbors(5, Distances.Cityblock())) isa RT
            @test knn_search(data, DescentNeighbors(5, Distances.Chebyshev())) isa RT

        end

        @testset "DescentNeighbors - Edge Cases" begin
            data = [rand(10) for _ in 1:100]
            RT = Tuple{Array{Int,2},Array{Float64,2}}

            # Small k
            @test knn_search(data, DescentNeighbors(1, Distances.Euclidean())) isa RT
            @test knn_search(data, DescentNeighbors(2, Distances.Euclidean())) isa RT

            # Large k (approaching n_points)
            @test knn_search(data, DescentNeighbors(50, Distances.Euclidean())) isa RT
            @test knn_search(data, DescentNeighbors(99, Distances.Euclidean())) isa RT

            # Small dataset
            small_data = [rand(5) for _ in 1:10]
            @test knn_search(small_data, DescentNeighbors(3, Distances.Euclidean())) isa RT
        end

        @testset "DescentNeighbors - Data Format Support" begin
            n, d, k = 50, 8, 5

            # Vector of vectors
            data_vecs = [rand(d) for _ in 1:n]
            knns_v, dists_v = knn_search(data_vecs, DescentNeighbors(k, Distances.Euclidean()))
            @test size(knns_v) == (k, n)

            # Matrix (columns are points)
            data_mat = rand(d, n)
            knns_m, dists_m = knn_search(data_mat, DescentNeighbors(k, Distances.Euclidean()))
            @test size(knns_m) == (k, n)

            # Both formats should work with nndescent
            @test knns_v isa Matrix{Int}
            @test knns_m isa Matrix{Int}
        end

        @testset "PrecomputedNeighbors - Distance Matrix" begin
            dist_mat = [0. 2. 1.;
                        2. 0. 3.;
                        1. 3. 0.]
            true_knns = [3 1 1;
                         2 3 2]
            true_dists = [1 2 1;
                          2 3 3]
            knn_params = PrecomputedNeighbors(2, dist_mat)

            @inferred knn_search(nothing, knn_params)
            knns, dists = knn_search(nothing, knn_params)
            @test knns == true_knns
            @test dists == true_dists

            # Named tuple test
            @inferred knn_search((view=nothing,), (view=knn_params,))
            res = knn_search((view=nothing,), (view=knn_params,))
            @test true_knns == res.view[1]
            @test true_dists == res.view[2]
        end

        @testset "PrecomputedNeighbors - Properties and Edge Cases" begin
            # Test with different k values
            dist_mat = [0. 5. 3. 1.;
                        5. 0. 2. 4.;
                        3. 2. 0. 6.;
                        1. 4. 6. 0.]

            # k=1: closest neighbor only
            knns_1, dists_1 = knn_search(nothing, PrecomputedNeighbors(1, dist_mat))
            @test size(knns_1) == (1, 4)
            @test knns_1 == [4 3 2 1]  # Closest to each point
            @test dists_1 == [1. 2. 2. 1.]

            # k=2
            knns_2, dists_2 = knn_search(nothing, PrecomputedNeighbors(2, dist_mat))
            @test size(knns_2) == (2, 4)
            # First row should be same as k=1
            @test knns_2[1, :] == knns_1[1, :]

            # k=3 (all neighbors except self)
            knns_3, dists_3 = knn_search(nothing, PrecomputedNeighbors(3, dist_mat))
            @test size(knns_3) == (3, 4)

            # Distances should be sorted within each column
            for col in eachcol(dists_3)
                @test issorted(col)
            end
        end

        @testset "PrecomputedNeighbors - Non-symmetric Distance Matrix" begin
            # Test with non-symmetric (directed) distances
            dist_mat = [0. 1. 5.;
                        3. 0. 2.;
                        4. 6. 0.]

            knns, dists = knn_search(nothing, PrecomputedNeighbors(2, dist_mat))

            # For point 1 (column 1): closest are points 2 (dist 3) and 3 (dist 4)
            @test knns[:, 1] == [2, 3]
            @test dists[:, 1] == [3., 4.]

            # For point 2 (column 2): closest are points 3 (dist 6) and 1 (dist 1)
            @test knns[:, 2] == [1, 3]
            @test dists[:, 2] == [1., 6.]
        end

        @testset "PrecomputedNeighbors - KNN Graph" begin
            # Test that we can extract neighbors from a graph correctly
            data = [rand(8) for _ in 1:50]
            k = 7

            # Build graph with descent
            descent_params = DescentNeighbors(k, Distances.SqEuclidean())
            knns_orig, dists_orig = knn_search(data, descent_params)

            # Wrap in PrecomputedNeighbors
            knn_graph = NND.HeapKNNGraph(data, descent_params.metric, knns_orig, dists_orig)
            precomp_params = PrecomputedNeighbors(k, knn_graph)

            # Extract should give identical results
            knns_extract, dists_extract = knn_search(data, precomp_params)
            @test knns_extract == knns_orig
            @test dists_extract == dists_orig
            @test size(knns_extract) == (k, 50)
        end

    end

    # -------------------------------------------------------------------------
    # FIT: Multi-View Tests
    # -------------------------------------------------------------------------

    @testset "FIT: Multi-View" begin

        @testset "Multiple Views - Mixed Neighbor Types" begin
            # This just tests that multi view returns the 
            # correct types, and that precomputed neighbors
            # still work.
            data = [rand(10) for _ in 1:100]
            desc_params = DescentNeighbors(5, Distances.Euclidean())
            RT = Tuple{Array{Int,2},Array{Float64,2}}
            dist_mat = [0. 2. 1.;
                        2. 0. 3.;
                        1. 3. 0.]
            true_knns = [3 1 1;
                         2 3 2]
            true_dists = [1 2 1;
                          2 3 3]
            pre_params = PrecomputedNeighbors(2, dist_mat)
            views_data = (view1=data, view2=nothing)
            views_knn_params = (view1=desc_params, view2=pre_params)

            @inferred knn_search(views_data, views_knn_params)
            views_results = knn_search(views_data, views_knn_params)
            @test views_results.view1 isa RT
            @test true_knns == views_results.view2[1]
            @test true_dists == views_results.view2[2]
        end

        @testset "Multiple Views - Consistent Keys" begin
            data1 = [rand(5) for _ in 1:30]
            data2 = [rand(8) for _ in 1:30]

            params1 = DescentNeighbors(4, Distances.Euclidean())
            params2 = DescentNeighbors(6, Distances.Cityblock())

            # Test with named tuples
            views_data = (view_a=data1, view_b=data2)
            views_params = (view_a=params1, view_b=params2)

            results = knn_search(views_data, views_params)

            # Check keys are preserved
            @test keys(results) == (:view_a, :view_b)

            # Check each result
            @test results.view_a isa Tuple{Array{Int,2}, Array{Float64,2}}
            @test results.view_b isa Tuple{Array{Int,2}, Array{Float64,2}}

            # Check shapes
            @test size(results.view_a[1]) == (4, 30)
            @test size(results.view_b[1]) == (6, 30)
        end

        @testset "Multiple Views - Different k Values" begin
            data = [rand(10) for _ in 1:50]

            # Same data, different k and metrics
            params_3 = DescentNeighbors(3, Distances.Euclidean())
            params_7 = DescentNeighbors(7, Distances.SqEuclidean())
            params_10 = DescentNeighbors(10, Distances.Cityblock())

            views_data = (v1=data, v2=data, v3=data)
            views_params = (v1=params_3, v2=params_7, v3=params_10)

            results = knn_search(views_data, views_params)

            # Each view should have different number of neighbors
            # the length of each point's neighbor list
            @test size(results.v1[1], 1) == 3
            @test size(results.v2[1], 1) == 7
            @test size(results.v3[1], 1) == 10

            # All should have same number of points
            @test size(results.v1[1], 2) == 50
            @test size(results.v2[1], 2) == 50
            @test size(results.v3[1], 2) == 50
        end

    end

    # -------------------------------------------------------------------------
    # TRANSFORM: Query Search Tests
    # -------------------------------------------------------------------------

    @testset "TRANSFORM: Query Search" begin

        @testset "DescentNeighbors - Basic Transform" begin
            # EXISTING TEST (kept for reference)
            data = [rand(10) for _ in 1:100]
            knn_params = DescentNeighbors(5, Distances.Euclidean())
            knns_dists = knn_search(data, knn_params)
            queries = [rand(10) for _ in 1:10]
            RT = Tuple{Array{Int,2},Array{Float64,2}}

            @inferred knn_search(data, queries, knn_params, knns_dists)
            query_knns_dists = knn_search(data, queries, knn_params, knns_dists)
            @test query_knns_dists isa RT
        end

        @testset "DescentNeighbors - Transform Output Properties" begin
            data = [rand(8) for _ in 1:100]
            queries = [rand(8) for _ in 1:15]
            knn_params = DescentNeighbors(7, Distances.Euclidean())

            # Fit
            knns_dists = knn_search(data, knn_params)

            # Transform
            query_knns, query_dists = knn_search(data, queries, knn_params, knns_dists)

            # Shape: k × n_queries
            @test size(query_knns) == (7, 15)
            @test size(query_dists) == (7, 15)

            # Indices should reference data points (1:100)
            @test all(1 .<= query_knns .<= 100)

            # Distances should be non-negative, in this case
            @test all(query_dists .>= 0)

            # Each query should have unique neighbors
            for col in eachcol(query_knns)
                @test length(unique(col)) == 7
            end
        end

        @testset "DescentNeighbors - Matrix Format Transform" begin
            data = [rand(10) for _ in 1:100]
            data_mat = rand(10, 100)
            queries_mat = rand(10, 10)
            knn_params = DescentNeighbors(5, Distances.Euclidean())
            knns_dists = knn_search(data, knn_params)
            RT = Tuple{Array{Int,2},Array{Float64,2}}

            knns, dists = knn_search(data_mat, queries_mat, knn_params, knns_dists)
            @test (knns, dists) isa RT
            @test size(knns) == (5, 10)
        end

        @testset "DescentNeighbors - Transform with Different Metrics" begin
            data = [rand(6) for _ in 1:50]
            queries = [rand(6) for _ in 1:8]

            for metric in [Distances.Euclidean(), Distances.SqEuclidean(), Distances.Cityblock(), Distances.Chebyshev()]
                knn_params = DescentNeighbors(5, metric)
                knns_dists = knn_search(data, knn_params)

                query_knns, query_dists = knn_search(data, queries, knn_params, knns_dists)
                @test size(query_knns) == (5, 8)
                @test all(query_dists .>= 0)
            end
        end

        @testset "Transform - Single View in NamedTuple" begin
            data = [rand(10) for _ in 1:100]
            queries = [rand(10) for _ in 1:10]
            knn_params = DescentNeighbors(5, Distances.Euclidean())
            knns_dists = knn_search(data, knn_params)
            RT = Tuple{Array{Int,2},Array{Float64,2}}

            @inferred knn_search((view1=data,), (view1=queries,), (view1=knn_params,), (view1=knns_dists,))
            query_knns_dists = knn_search((view1=data,), (view1=queries,), (view1=knn_params,), (view1=knns_dists,))
            @test query_knns_dists.view1 isa RT
        end

        @testset "Transform - Multi-View NamedTuple" begin
            data1 = [rand(5) for _ in 1:40]
            data2 = [rand(8) for _ in 1:40]
            queries1 = [rand(5) for _ in 1:10]
            queries2 = [rand(8) for _ in 1:10]

            params1 = DescentNeighbors(4, Distances.Euclidean())
            params2 = DescentNeighbors(6, Distances.Cityblock())

            # Fit on each view
            knns_dists1 = knn_search(data1, params1)
            knns_dists2 = knn_search(data2, params2)

            # Transform with NamedTuple
            views_data = (v1=data1, v2=data2)
            views_queries = (v1=queries1, v2=queries2)
            views_params = (v1=params1, v2=params2)
            views_knns_dists = (v1=knns_dists1, v2=knns_dists2)

            query_results = knn_search(views_data, views_queries, views_params, views_knns_dists)

            # Check keys preserved
            @test keys(query_results) == (:v1, :v2)

            # Check shapes
            @test size(query_results.v1[1]) == (4, 10)
            @test size(query_results.v2[1]) == (6, 10)
        end

        @testset "PrecomputedNeighbors - Transform with Distance Matrix" begin
            dist_mat = [0. 2. 1.;
                        2. 0. 3.;
                        1. 3. 0.]
            knns_dists = knn_search(nothing, PrecomputedNeighbors(2, dist_mat))
            query_dist_mat = [1. 2.;
                              3. 3.;
                              2. 1.]
            query_true_knns = [1 3;
                               3 1;
                               2 2]
            query_true_dists = [1. 1.;
                                2. 2.;
                                3. 3.]

            @inferred knn_search(nothing, nothing, PrecomputedNeighbors(3, query_dist_mat), knns_dists)
            query_knns_dists = knn_search(nothing, nothing, PrecomputedNeighbors(3, query_dist_mat), knns_dists)
            @test query_true_knns == query_knns_dists[1]
            @test query_true_dists == query_knns_dists[2]
        end
    end

    # -------------------------------------------------------------------------
    # Helper Function Tests: _knn_from_dists
    # -------------------------------------------------------------------------

    @testset "Helper: _knn_from_dists" begin

        @testset "_knn_from_dists - Basic Functionality" begin
            # Simple 3x3 symmetric distance matrix
            dist_mat = [0. 2. 1.;
                        2. 0. 3.;
                        1. 3. 0.]

            # k=1, ignore_diagonal=true (default)
            knns, dists = _knn_from_dists(dist_mat, 1)
            @test knns == [3 1 1]
            @test dists == [1. 2. 1.]

            # k=2, ignore_diagonal=true
            knns, dists = _knn_from_dists(dist_mat, 2)
            @test size(knns) == (2, 3)
            @test size(dists) == (2, 3)
        end

        @testset "_knn_from_dists - ignore_diagonal Behavior" begin
            dist_mat = [0. 5. 3.;
                        5. 0. 2.;
                        3. 2. 0.]

            # With ignore_diagonal=true (skip the 0 on diagonal)
            knns_ignore, dists_ignore = _knn_from_dists(dist_mat, 1, ignore_diagonal=true)
            # Point 1: closest non-self is point 3 (dist 3)
            @test knns_ignore[:, 1] == [3]
            @test dists_ignore[:, 1] == [3.]

            # With ignore_diagonal=false (include the 0 on diagonal)
            knns_include, dists_include = _knn_from_dists(dist_mat, 1, ignore_diagonal=false)
            # Point 1: closest is itself (dist 0)
            @test knns_include[:, 1] == [1]
            @test dists_include[:, 1] == [0.]
        end

        @testset "_knn_from_dists - Non-square Matrix (Transform)" begin
            # 3 data points, 2 queries: 2×3 matrix
            # Rows are queries, columns are data points
            query_dist_mat = [3. 1. 2.;
                              4. 5. 1.]

            # ignore_diagonal should be false for non-square matrices
            knns, dists = _knn_from_dists(query_dist_mat, 2, ignore_diagonal=false)

            # Query 1 (column 1): nearest in col 1 are indices 1 (dist 3) and 2 (dist 4)
            @test knns[:, 1] == [1, 2]
            @test dists[:, 1] == [3., 4.]

            # Query 2 (column 2): nearest in col 3 are indices 2 (dist 1) and 1 (dist 2)
            @test knns[:, 3] == [2, 1]
            @test dists[:, 3] == [1., 2.]
        end

        @testset "_knn_from_dists - Sorted Output" begin
            dist_mat = [0. 9. 3. 1. 7.;
                        9. 0. 5. 8. 2.;
                        3. 5. 0. 4. 6.;
                        1. 8. 4. 0. 10.;
                        7. 2. 6. 10. 0.]

            knns, dists = _knn_from_dists(dist_mat, 3, ignore_diagonal=true)

            # For each column, distances should be sorted (increasing)
            for col in eachcol(dists)
                @test issorted(col)
            end

            # For each column, indices should correspond to sorted distances
            for i in axes(dist_mat, 2)
                @test dists[:, i] == dist_mat[knns[:, i], i]
            end
        end

        @testset "_knn_from_dists - Edge Cases" begin
            # 2×2 matrix, k=1
            small_mat = [0. 1.;
                         1. 0.]
            knns, dists = _knn_from_dists(small_mat, 1, ignore_diagonal=true)
            @test knns == [2 1]
            @test dists == [1. 1.]

            # Single column (1 data point with self-distance)
            single_col = reshape([0.], 1, 1)
            knns, dists = _knn_from_dists(single_col, 1, ignore_diagonal=false)
            @test knns == reshape([1], 1, 1)
            @test dists == reshape([0.], 1, 1)

            # Request all neighbors (k = n-1 with ignore_diagonal=true)
            mat_4 = [0. 1. 2. 3.;
                     1. 0. 4. 5.;
                     2. 4. 0. 6.;
                     3. 5. 6. 0.]
            knns, dists = _knn_from_dists(mat_4, 3, ignore_diagonal=true)
            @test size(knns) == (3, 4)
            # Point 1 should have neighbors 2, 3, 4
            @test Set(knns[:, 1]) == Set([2, 3, 4])
        end

        @testset "_knn_from_dists - Type Preservation" begin
            # Test that element type is preserved
            dist_mat_f32 = Float32[0. 1. 2.;
                                   1. 0. 3.;
                                   2. 3. 0.]
            knns, dists = _knn_from_dists(dist_mat_f32, 1)
            @test eltype(dists) == Float32
            @test eltype(knns) == Int

            dist_mat_f64 = Float64[0. 1. 2.;
                                   1. 0. 3.;
                                   2. 3. 0.]
            knns, dists = _knn_from_dists(dist_mat_f64, 1)
            @test eltype(dists) == Float64
        end

    end

end
