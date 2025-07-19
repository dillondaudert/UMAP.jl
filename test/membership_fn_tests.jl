
@testset "membership_fn tests" begin

    @testset "fit_ab function tests" begin
        
        @testset "basic functionality" begin
            # Test with default UMAP parameters
            a, b = UMAP.fit_ab(0.1, 1.0)
            @test a > 0
            @test b isa Real
            @test isfinite(a) && isfinite(b)
        end

        @testset "parameter variations" begin
            # Test different min_dist values
            a1, b1 = UMAP.fit_ab(0.01, 1.0)
            a2, b2 = UMAP.fit_ab(0.1, 1.0)
            a3, b3 = UMAP.fit_ab(0.5, 1.0)
            
            @test all(isfinite.([a1, b1, a2, b2, a3, b3]))
            @test all([a1, a2, a3] .> 0)
            
            # Test different spread values
            a4, b4 = UMAP.fit_ab(0.1, 0.5)
            a5, b5 = UMAP.fit_ab(0.1, 2.0)
            
            @test all(isfinite.([a4, b4, a5, b5]))
            @test all([a4, a5] .> 0)
        end

        @testset "edge cases" begin
            # Very small min_dist
            a, b = UMAP.fit_ab(1e-6, 1.0)
            @test a > 0 && isfinite(a)
            @test isfinite(b)
            
            # Very small spread
            a, b = UMAP.fit_ab(0.1, 1e-3)
            @test a > 0 && isfinite(a)
            @test isfinite(b)
            
            # Large values
            a, b = UMAP.fit_ab(2.0, 5.0)
            @test a > 0 && isfinite(a)
            @test isfinite(b)
        end

        @testset "mathematical properties" begin
            min_dist, spread = 0.1, 1.0
            a, b = UMAP.fit_ab(min_dist, spread)
            
            # Test the fitted curve approximates the target function
            curve(x, a, b) = (1 + a * x^(2*b))^(-1)
            target(d) = d >= min_dist ? exp(-(d - min_dist)/spread) : 1.0
            
            # Test at a few key points
            test_points = [0.0, min_dist/2, min_dist, min_dist + spread, min_dist + 2*spread]
            
            for x in test_points
                if x >= 0
                    fitted_val = curve(x, a, b)
                    target_val = target(x)
                    # Should be reasonably close (within 20% for this approximation)
                    @test abs(fitted_val - target_val) / target_val < 0.2
                end
            end
        end

        @testset "consistency tests" begin
            # Same parameters should give same results
            a1, b1 = UMAP.fit_ab(0.1, 1.0)
            a2, b2 = UMAP.fit_ab(0.1, 1.0)
            @test a1 ≈ a2 atol=1e-10
            @test b1 ≈ b2 atol=1e-10
            
            # Test type stability
            result = @inferred UMAP.fit_ab(0.1, 1.0)
            @test result isa Tuple{Float64, Float64}
        end

        @testset "boundary behavior" begin
            # When min_dist approaches spread
            a1, b1 = UMAP.fit_ab(0.9, 1.0)
            @test a1 > 0 && isfinite(a1)
            @test isfinite(b1)
            
            # When min_dist equals spread
            a2, b2 = UMAP.fit_ab(1.0, 1.0)
            @test a2 > 0 && isfinite(a2)
            @test isfinite(b2)
        end
    end

end