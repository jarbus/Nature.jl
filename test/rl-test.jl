
using Infiltrator
@testset verbose = true "Test RL Loop" begin

    MAX_STEPS=10
    N_STARTING_PLAYERS = 2
    UPDATE_FREQ = 6
    WORLD_SIZE = (12, 12)
    clip = 0.1f0

    env = NatureEnv(num_starting_players=N_STARTING_PLAYERS,
                    world_size=WORLD_SIZE,
                    episode_len=UPDATE_FREQ+2,
                    num_frames=4,
                    window=3,
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

    @testset "Sane MultiPPO states" begin
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
        @test is_terminated(env)
        # check you can create terminal state
        @test [state(env, p) for p in 1:length(env.players)] !== nothing
    end

    @testset "Frame tests" begin
        nfc = Int(env.obs_size[3] / env.num_frames) # num frame channels

        # Timestep 1
        # Test initial zero frames are equal
        @test all(0f0 .== states[1][:,:,1:3*nfc, :])
        @test states[1][:,:,1:nfc,:] == states[1][:,:,nfc+1:2nfc,:] == states[1][:,:,2nfc+1:3nfc,:]
        # Test initial zero frame is not equal to starting frame
        @test states[1][:,:,2nfc+1:3nfc,:] != states[1][:,:,3nfc+1:end,:]

        # Timestep 2
        @test states[2][:,:,1:nfc,:] == states[2][:,:,nfc+1:2nfc,:] != states[2][:,:,2nfc+1:3nfc,:]
        @test states[2][:,:,2nfc+1:3nfc,:] != states[2][:,:,3nfc+1:end,:]
        @test states[1][:,:,3nfc+1:end,:] == states[2][:,:,2nfc+1:3nfc,:]

        # Timestep 3
        @test all(0f0 .== states[3][:,:,1:nfc,:])
        @test states[3][:,:,1:nfc,:] != states[3][:,:,nfc+1:2nfc,:]
        @test states[3][:,:,1:nfc,:] != states[3][:,:,nfc+1:2nfc,:]
        @test states[3][:,:,2nfc+1:3nfc,:] != states[3][:,:,3nfc+1:end,:]
        @test states[2][:,:,3nfc+1:end,:] == states[3][:,:,2nfc+1:3nfc,:]

        # Timestep 4
        @test states[1][:,:,3nfc+1:end,:] == states[4][:,:,1:nfc,:]
        @test states[2][:,:,3nfc+1:end,:] == states[4][:,:,1nfc+1:2nfc,:]
        @test states[3][:,:,3nfc+1:end,:] == states[4][:,:,2nfc+1:3nfc,:]
        @test states[4][:,:,2nfc+1:3nfc,:] != states[4][:,:,3nfc+1:end,:]

        # Timestep 5
        @test states[1][:,:,3nfc+1:end,:] == states[4][:,:,1:nfc,:]
        @test states[2][:,:,3nfc+1:end,:] == states[4][:,:,1nfc+1:2nfc,:]
        @test states[3][:,:,3nfc+1:end,:] == states[4][:,:,2nfc+1:3nfc,:]
        @test states[4][:,:,2nfc+1:3nfc,:] != states[4][:,:,3nfc+1:end,:]
        # Last timestep
        @test states[5][:,:,3nfc+1:end,:] == states[8][:,:,1:nfc,:]
        @test states[6][:,:,3nfc+1:end,:] == states[8][:,:,1nfc+1:2nfc,:]
        @test states[7][:,:,3nfc+1:end,:] == states[8][:,:,2nfc+1:3nfc,:]
        @test states[8][:,:,2nfc+1:3nfc,:] != states[8][:,:,3nfc+1:end,:]
        step()
        @test env.step == env.episode_len == UPDATE_FREQ + 2
    end
end
