
using Infiltrator
@testset verbose = true "Test MultiPPO" begin

    MAX_STEPS=10
    N_STARTING_PLAYERS = 2
    UPDATE_FREQ = 6
    WORLD_SIZE = (12, 12)
    clip = 0.1f0

    env = NatureEnv(num_starting_players=N_STARTING_PLAYERS,
                    world_size=WORLD_SIZE,
                    max_step=UPDATE_FREQ+1,
                    food_generators=[
                        FoodGen([4,4],[30,30], max_pos=WORLD_SIZE[1:2]),
                       ])
    RLBase.reset!(env)
    ns, na = size(state(env, 1)), length(action_space(env,1))

    policy = build_MultiPPOManager(env, N_STARTING_PLAYERS, UPDATE_FREQ, clip)


    policy(PRE_EXPERIMENT_STAGE, env)
    policy(PRE_EPISODE_STAGE, env)
    acts = []
    states = []
    rews = []

    function step()
        s = [state(env, p) for p in 1:length(env.players)]
        s = cat(s..., dims=ndims(s[1])+1)

        action = policy(env)
        policy(PRE_ACT_STAGE, env, action)
        env(action)
        policy(POST_ACT_STAGE, env)


        a = [action[p].action for p in 1:length(env.players)]
        a = cat(a..., dims=ndims(a[1])+1)

        r = [reward(env, p) for p in 1:length(env.players)]
        r = cat(r..., dims=ndims(r[1])+1)

        push!(states, s)
        push!(acts, a)
        push!(rews, r)
    end

    for i in 1:UPDATE_FREQ
        step()
        @test states[i] == policy.agents[1].trajectory[:state][:,:,:,:,i]
    end
    @test states[1] == policy.agents[1].trajectory[:state][:,:,:,:,1]
    @test states[2] == policy.agents[1].trajectory[:state][:,:,:,:,2]
    @test states[3] == policy.agents[1].trajectory[:state][:,:,:,:,3]
    # State buffer is one size bigger, so size 3 will be size 4
    step()
    step()
    @test states[2] == policy.agents[1].trajectory[:state][:,:,:,:,1]
    @test states[3] == policy.agents[1].trajectory[:state][:,:,:,:,2]
    @test states[4] == policy.agents[1].trajectory[:state][:,:,:,:,3]
end
