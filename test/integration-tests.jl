@testset verbose = true "Integration Tests" begin

    @testset "Clear board" begin
        # Write your tests here.
        env = NatureEnv(
            num_starting_players=2,
            world_size=(8, 8),
            window=0,
            player_starting_food=100f0,
            food_generators=[
                FoodGen([5,5],[1,1], max_pos=(8,8)),
                FoodGen([2,2],[1,1], max_pos=(8,8)),
               ])

        RLBase.reset!(env)
        Nature.move_and_collect(env, 1, 1)
        Nature.move_and_collect(env, 2, 2)
        @test env.food_frames[1:2] |> sum |> unique == [0]
    end


    @testset "Pick/Place" begin
        env = NatureEnv(
            num_starting_players=1,
            world_size=(1, 1),
            window=0,
            food_generators=[
                FoodGen([1,1],[1,1], max_pos=(1,1)),
               ])

        RLBase.reset!(env)
        @test env.players[1].pos == (1,1)
        @test env.players[1].food_counts == (env.player_starting_food,)
        f = env.food_frames[1][env.players[1].pos...]
        env(PICK_FOOD_1,1)
        @test env.players[1].food_counts[1] == env.player_starting_food + f - 0.1f0
        env(PLACE_FOOD_1,1)
        @test 0.1f0 > abs(env.players[1].food_counts[1] - (env.player_starting_food + f - 0.2f0 - 0.5f0))
    end

    @testset "Exchange" begin
        env = NatureEnv(
            num_starting_players=2,
            world_size=(1, 1),
            window=0,
            food_generators=[
                FoodGen([1,1],[1,1], max_pos=(1,1)),
               ])

        RLBase.reset!(env)
        ff = env.food_frames[1][1,1]
        env(PLACE_FOOD_1, 1)
        @test env.food_frames[1][1,1] == ff + 0.5f0
        env(PICK_FOOD_1, 2)
        @test env.exchanges[1] == 0.5f0
    end
end
