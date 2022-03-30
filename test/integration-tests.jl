
@testset verbose = true "Integration Tests" begin

    @testset "Clear board" begin
        # Write your tests here.
        nenv = NatureEnv(
            num_starting_players=2,
            world_size=(32, 32, 4),
            food_generators=[
                FoodGen([5,5],[1,1]),
                FoodGen([25,25],[1,1]),
               ])

        Nature.reset!(nenv)
        Nature.move_and_collect(nenv, 1, 1)
        Nature.move_and_collect(nenv, 2, 2)
        @test nenv.food_frames[1:2] |> sum |> unique == [0]
        @test Nature.is_terminated(nenv) == true
    end


    @testset "Test Actions on Small Board" begin
        nenv = NatureEnv(
            num_starting_players=1,
            world_size=(1, 1, 3),
            food_generators=[
                FoodGen([1,1],[1,1], max_pos=(1,1)),
               ])

        Nature.reset!(nenv)
        @test nenv.players[1].pos == (1,1)
        @test nenv.players[1].food_counts == (0,)
        nenv(5,1)
        @test nenv.players[1].food_counts[1] >= 1
        nenv(6,1)
        @test nenv.players[1].food_counts == (0,)

    end
end
