function build_MultiPPOManager(env::NatureEnv, num_starting_players::Int, update_freq::Int, clip::Float32)
    ns, na = size(state(env, 1)), length(action_space(env,1))

    cnn_output_shape = Int.(floor.([ns[1], ns[2], 64]))
    create_actor() = Chain(
        Conv((5,5), ns[3]=>64, pad=(2,2), relu),
        # MaxPool((2,2)),
        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 64=>64, pad=(1,1), relu),
        # MaxPool((2,2)),
        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 64=>64, pad=(1,1), relu),
        # MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_shape), 1024),
        Dense(1024, 1024, relu),
        Dense(1024, 512, relu),
        Dense(512, na, relu)
    )

    create_critic() = Chain(
        Conv((5,5), ns[3]=>64, pad=(2,2), relu),
        # MaxPool((2,2)),
        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 64=>64, pad=(1,1), relu),
        # MaxPool((2,2)),
        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 64=>64, pad=(1,1), relu),
        # MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_shape), 1024),
        Dense(1024, 1024, relu),
        Dense(1024, 512, relu),
        Dense(512, 1, relu)
    )

    create_trajectory() = PPOTrajectory(;
        capacity = update_freq,
        state = Array{Float32, 4} => (ns..., num_starting_players),
        action = Vector{Int} => (num_starting_players,),
        action_log_prob = Vector{Float32} => (num_starting_players,),
        reward = Vector{Float32} => (num_starting_players,),
        terminal = Vector{Bool} => (num_starting_players,),
    )

    # TODO CHANGE THIS TO BE DYNAMIC UPDATE FREQ AS NUM AGENTS CHANGES
    create_policy() = PPOPolicy(
        approximator = ActorCritic(
            actor = create_actor() |> gpu,
            critic = create_critic() |> gpu,
            optimizer = ADAM(1e-3),
        ),
        γ = 0.99f0,
        λ = 0.95f0,
        clip_range = clip,
        max_grad_norm = 0.5f0,
        n_epochs = 4,
        n_microbatches = 4,
        actor_loss_weight = 1.0f0,
        critic_loss_weight = 0.5f0,
        entropy_loss_weight = 0.001f0,
        update_freq = update_freq,
    )

    create_agent() = Agent(
        policy = create_policy(),
        trajectory = create_trajectory()
    )
    agent1 = create_agent()
    agent_map = Dict(player => agent1 for player in 1:num_starting_players)
    agent_inv_map = Dict(agent1 => [player for player in 1:num_starting_players])

    agents = MultiPPOManager(
        agent_map,
        agent_inv_map,
        PPOTrajectory, # trace's type
    )
end
