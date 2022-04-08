@testset verbose = true "Integration Tests" begin

    # @testset "Clear board" begin
    #     # Write your tests here.
    #     env = NatureEnv(
    #         num_starting_players=2,
    #         world_size=(8, 8),
    #         window=0,
    #         player_starting_food=100f0,
    #         food_generators=[
    #             FoodGen([5,5],[1,1], max_pos=(8,8)),
    #             FoodGen([2,2],[1,1], max_pos=(8,8)),
    #            ])

    #     RLBase.reset!(env)
    #     Nature.move_and_collect(env, 1, 1)
    #     Nature.move_and_collect(env, 2, 2)
    #     @test env.food_frames[1:2] |> sum |> unique == [0]
    # end


    # @testset "Pick/Place" begin
    #     env = NatureEnv(
    #         num_starting_players=1,
    #         world_size=(1, 1),
    #         window=0,
    #         food_generators=[
    #             FoodGen([1,1],[1,1], max_pos=(1,1)),
    #            ])

    #     RLBase.reset!(env)
    #     @test env.players[1].pos == (1,1)
    #     @test env.players[1].food_counts == (env.player_starting_food,)
    #     f = env.food_frames[1][env.players[1].pos...]
    #     env(PICK_FOOD_1,1)
    #     @test env.players[1].food_counts[1] == env.player_starting_food + f - 0.1f0
    #     env(PLACE_FOOD_1,1)
    #     @test 0.1f0 > abs(env.players[1].food_counts[1] - (env.player_starting_food + f - 0.2f0 - 0.5f0))
    # end

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
        # Test exchange works between agents
        env(PLACE_FOOD_1, 1)
        @test env.food_frames[1][1,1] == ff + 0.5f0
        env(PICK_FOOD_1, 2)
        @test env.exchanges[1] == 0.5f0
        # Test exchange doesn't work between same agent
        env(PLACE_FOOD_1, 1)
        env(PICK_FOOD_1, 1)
        @test env.exchanges[1] == 0.5f0
        RLBase.reset!(env)
        @test sum(env.exchanges) == 0f0
    end

    @testset "Comm" begin
        env = NatureEnv(
            num_starting_players=2,
            world_size=(1, 1),
            window=0,
            vocab_size=2,
            food_generators=[
                FoodGen([1,1],[1,1], max_pos=(1,1)),
               ])

        RLBase.reset!(env)
        COMM_1 = 4+(2*env.food_types) + 1
        COMM_1_CHANNEL = 7
        COMM_2 = 4+(2*env.food_types) + 2
        COMM_2_CHANNEL = 8
        @test state(env, 1)[:,:,COMM_1_CHANNEL][1] == state(env, 2)[:, :, COMM_1_CHANNEL][1] == 0f0
        @test state(env, 1)[:,:,COMM_2_CHANNEL][1] == state(env, 2)[:, :, COMM_2_CHANNEL][1] == 0f0
        env(COMM_1, 1)
        @test state(env, 1)[:,:,COMM_1_CHANNEL][1] == state(env, 2)[:, :, COMM_1_CHANNEL][1] == 1f0
        @test state(env, 1)[:,:,COMM_2_CHANNEL][1] == state(env, 2)[:, :, COMM_2_CHANNEL][1] == 0f0
        env(COMM_2, 2)
        @test state(env, 1)[:,:,COMM_1_CHANNEL][1] == state(env, 2)[:, :, COMM_1_CHANNEL][1] == 1f0
        @test state(env, 1)[:,:,COMM_2_CHANNEL][1] == state(env, 2)[:, :, COMM_2_CHANNEL][1] == 1f0
        env.step += 1
        @test state(env, 1)[:,:,COMM_1_CHANNEL][1] == state(env, 2)[:, :, COMM_1_CHANNEL][1] == 0f0
        @test state(env, 1)[:,:,COMM_2_CHANNEL][1] == state(env, 2)[:, :, COMM_2_CHANNEL][1] == 0f0
        RLBase.reset!(env)
        @test length(env.comms) == env.episode_len
        @test all([isempty(comm_step) for comm_step in env.comms])
    end
end
