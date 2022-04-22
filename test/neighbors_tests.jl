
@testset "neighbors tests" begin

    @testset "fit tests" begin
        @testset "single view tests" begin
            @testset "descent neighbors" begin
                data = [rand(10) for _ in 1:100]
                knn_params = DescentNeighbors(5, Euclidean())
                knn_params_kw = DescentNeighbors(5, Euclidean(), (max_iters=5,))
                knn_params_bad_kw = DescentNeighbors(5, Euclidean(), (bad_kw=10.,))
                RT = Tuple{Array{Int,2},Array{Float64,2}}
                @inferred knn_search(data, knn_params)
                # non-named tuple test
                @test knn_search(data, knn_params) isa RT
                # kwargs test
                @test knn_search(data, knn_params_kw) isa RT
                @test_throws MethodError knn_search(data, knn_params_bad_kw)
                # named tuple test
                @inferred knn_search((view=data,), (view=knn_params,))
                res = knn_search((view=data,), (view=knn_params,))
                @test res.view isa RT
            end
            @testset "precomputed neighbors" begin
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
                # named tuple test
                @inferred knn_search((view=nothing,), (view=knn_params,))
                res = knn_search((view=nothing,), (view=knn_params,))
                @test true_knns == res.view[1]
                @test true_dists == res.view[2]
            end
        end
        @testset "multiple views tests" begin
            data = [rand(10) for _ in 1:100]
            desc_params = DescentNeighbors(5, Euclidean())
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

    end

    @testset "transform tests" begin
        # test basic transform knn_search
        data = [rand(10) for _ in 1:100]
        knn_params = DescentNeighbors(5, Euclidean())
        # get the result from the fit step
        knns_dists = knn_search(data, knn_params)
        queries = [rand(10) for _ in 1:10]
        RT = Tuple{Array{Int,2},Array{Float64,2}}
        @inferred knn_search(data, queries, knn_params, knns_dists)
        query_knns_dists = knn_search(data, queries, knn_params, knns_dists)
        @test query_knns_dists isa RT

        # as a matrix
        data_mat = rand(10, 100)
        queries_mat = rand(10, 10)
        knns_dists = knn_search(data, knn_params)
        @test knn_search(data_mat, queries_mat, knn_params, knns_dists) isa RT

        # views in named tuple test
        @inferred knn_search((view1=data,), (view1=queries,), (view1=knn_params,), (view1=knns_dists,))
        query_knns_dists = knn_search((view1=data,), (view1=queries,), (view1=knn_params,), (view1=knns_dists,))
        @test query_knns_dists.view1 isa RT

        # precomputed distances test 
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

        # test that 
    end

end
