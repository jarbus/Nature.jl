
@testset verbose = true "Integration Tests" begin

    @testset "Clear board" begin
        # Write your tests here.
        nenv = NatureEnv()
        Nature.reset!(nenv)
        Nature.move_and_collect(nenv, 1, 1)
        Nature.move_and_collect(nenv, 2, 2)
        @test nenv.food_frames[1:2] |> sum |> unique == [0]
        @test Nature.is_terminated(nenv) == true
    end
end
