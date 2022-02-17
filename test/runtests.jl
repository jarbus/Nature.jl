using Nature
using Test

@testset "Nature.jl" begin
    # Write your tests here.
    nenv = NatureEnv()
    @test nenv isa NatureEnv
    @test length(nenv.players) > 0
    for p in nenv.players
        @test length(p.food_counts) == nenv.food_types
    end

end
