using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Nature
# using Serialization
using TensorBoardLogger, Logging
using Infiltrator
using SparseArrays

MAX_STEPS=1_000_000


function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:Nature},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)


    N_STARTING_PLAYERS = 8
    UPDATE_FREQ = 200
    OBS_SIZE = (32, 32, 3)
    clip = 0.1f0

    lg=TBLogger("tensorboard_logs/run clip=$clip uf=$UPDATE_FREQ")
    global_logger(lg)


    rng = StableRNG(seed)
    env = NatureEnv(num_starting_players=N_STARTING_PLAYERS,
                    observation_size=OBS_SIZE,
                    food_generators=[
                        FoodGen([15,15],[20,20]),
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

    # TODO CHANGE THIS TO BE DYNAMIC UPDATE FREQ AS NUM AGENTS CHANGES
    function create_policy()
        PPOPolicy(
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
            update_freq = UPDATE_FREQ*N_STARTING_PLAYERS,
        )
    end

    create_agent() = Agent(
        policy = create_policy(),
        trajectory = create_trajectory()
    )
    agent1 = create_agent()
    agent_map = Dict(player => agent1 for player in 1:N_STARTING_PLAYERS)
    agent_inv_map = Dict(agent1 => [player for player in 1:N_STARTING_PLAYERS])

    agents = MultiPPOManager(
        agent_map,
        agent_inv_map,
        PPOTrajectory, # trace's type
        512, # batch_size
        rng
    )

    stop_condition = StopAfterStep(MAX_STEPS, is_show_progress=!haskey(ENV, "CI"))
    hook = NatureHook(env)
    Experiment(agents, env, stop_condition, hook, "# PPO with Nature")

end

ex = E`JuliaRL_PPO_Nature`
run(ex)
# step_through_env(ex.env, policy)
