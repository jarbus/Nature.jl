
@testset verbose = true "Unit Tests" begin

    @testset "Initialization" begin
        # Write your tests here.
        nenv = NatureEnv()
        @test nenv isa NatureEnv
        Nature.reset!(nenv)
        @test length(nenv.players) == nenv.num_starting_players
        for p in nenv.players
            @test length(p.food_counts) == nenv.food_types
        end
        @test length(nenv.food_frames) == nenv.food_types
        for frame in nenv.food_frames
            @test size(frame) == nenv.world_size[1:2]
        end
    end

    @testset "Overrides" begin
        nenv = NatureEnv()
        Nature.reset!(nenv)
        states = [Nature.state(nenv, i) for i in 1:length(nenv.players)]
        # Test states for all players are the same size
        @test size.(states) |> unique |> length == 1
        @test size(states[1])[1:2] == size(nenv) .+ nenv.window
        @test sum(Nature.reward(nenv, p) for p in 1:length(nenv.players)) == 0
        @test !any(Nature.is_terminated(nenv, i) for i in 1:length(nenv.players))
        [Nature.action_space(nenv, i) for i in 1:length(nenv.players)]
    end

end
