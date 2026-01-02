using UMAP: MembershipFnParams

@testset "membership_fn tests" begin

    @testset "MembershipFnParams tests" begin
        @test_throws ArgumentError MembershipFnParams(-1, 1)
        @test_throws ArgumentError MembershipFnParams(1, -.01)
        @test_throws ArgumentError MembershipFnParams(-1, 1, 1, 1)
        @test_throws ArgumentError MembershipFnParams(1, -0.1, 1, 1)

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
    end

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