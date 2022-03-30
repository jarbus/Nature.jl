
@testset verbose = true "Test MultiPPO" begin

    MAX_STEPS=10
    N_STARTING_PLAYERS = 2
    UPDATE_FREQ = 3
    WORLD_SIZE = (8, 8, 3)
    clip = 0.1f0

    env = NatureEnv(num_starting_players=N_STARTING_PLAYERS,
                    world_size=WORLD_SIZE,
                    food_generators=[
                        FoodGen([4,4],[30,30], max_pos=WORLD_SIZE[1:2]),
                       ])
    Nature.reset!(env)
    ns, na = size(state(env, 1)), length(action_space(env,1))

    cnn_output_shape = Int.(floor.([ns[1]/8, ns[2]/8,32]))
    create_actor() = Chain(
        Conv((3,3), ns[3]=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_shape), 64),
        Dense(64, na)
    )

    create_critic() = Chain(
        Conv((3,3), ns[3]=>16, pad=(1,1), relu),
        MaxPool((2,2)),
        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_shape), 64),
        Dense(64, 1)
    )

    create_trajectory() = PPOTrajectory(;
        capacity = UPDATE_FREQ,
        state = Array{Float32, 4} => (ns..., N_STARTING_PLAYERS),
        action = Vector{Int} => (N_STARTING_PLAYERS,),
        action_log_prob = Vector{Float32} => (N_STARTING_PLAYERS,),
        reward = Vector{Float32} => (N_STARTING_PLAYERS,),
        terminal = Vector{Bool} => (N_STARTING_PLAYERS,),
    )

    function create_policy()
        PPOPolicy(
            approximator = ActorCritic(
                actor = create_actor(),
                critic = create_critic(),
                optimizer = ADAM(1e-3),
            ),
            γ = 0.99f0,
            λ = 0.95f0,
            clip_range = clip,
            max_grad_norm = 0.5f0,
            n_epochs = 4,
            n_microbatches = 2,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.001f0,
            update_freq = UPDATE_FREQ,
        )
    end

    create_agent() = Agent(
        policy = create_policy(),
        trajectory = create_trajectory()
    )
    agent1 = create_agent()
    agent_map = Dict(player => agent1 for player in 1:N_STARTING_PLAYERS)
    agent_inv_map = Dict(agent1 => [player for player in 1:N_STARTING_PLAYERS])

    policy = MultiPPOManager(
        agent_map,
        agent_inv_map,
        PPOTrajectory, # trace's type
    )

    reset!(env)
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
