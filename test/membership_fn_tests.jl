
@testset "membership_fn tests" begin

@testset "fit_ab tests" begin
@test all((1, 2) .== fit_ab(-1, 0, 1, 2))
end

end