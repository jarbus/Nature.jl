using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Nature
# using Serialization
using TensorBoardLogger, Logging
using Infiltrator
using Dates
using Serialization

MAX_STEPS=1_000_000

N_STARTING_PLAYERS = 36
UPDATE_FREQ = 64
WORLD_SIZE = (64, 64)
clip = 0.1f0

trial_id = ""

timestamp = Dates.format(now(),"yyyy-mm-dd HH:MM:SS")

function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:Nature},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)


    rng = StableRNG(seed)
    env = NatureEnv(num_starting_players=N_STARTING_PLAYERS,
                    world_size=WORLD_SIZE,
                    window=3,
                    food_generators=[
                        FoodGen([15,15],[60,60]),
                        FoodGen([45,45],[60,60]),
                       ])
    Nature.reset!(env)


    global trial_id = "$timestamp clip=$clip uf=$UPDATE_FREQ maxsteps=$MAX_STEPS ws=$(WORLD_SIZE[1:2]) nf=$(length(env.food_generators))"
    lg=TBLogger("tensorboard_logs/$trial_id")
    global_logger(lg)


    ns, na = size(state(env, 1)), length(action_space(env,1))

    cnn_output_shape = Int.(floor.([ns[1], ns[2],32]))
    create_actor() = Chain(
        Conv((3,3), ns[3]=>16, pad=(1,1), relu),
        # MaxPool((2,2)),
        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        # MaxPool((2,2)),
        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        # MaxPool((2,2)),
        flatten,
        Dense(prod(cnn_output_shape), 64),
        Dense(64, na)
    )

    create_critic() = Chain(
        Conv((3,3), ns[3]=>16, pad=(1,1), relu),
        # MaxPool((2,2)),
        # Second convolution, operating upon a 14x14 image
        Conv((3, 3), 16=>32, pad=(1,1), relu),
        # MaxPool((2,2)),
        # Third convolution, operating upon a 7x7 image
        Conv((3, 3), 32=>32, pad=(1,1), relu),
        # MaxPool((2,2)),
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

    agents = MultiPPOManager(
        agent_map,
        agent_inv_map,
        PPOTrajectory, # trace's type
    )

    stop_condition = StopAfterStep(MAX_STEPS, is_show_progress=!haskey(ENV, "CI"))
    hook = NatureHook(env)

    Experiment(agents, env, stop_condition, hook, "# PPO with Nature")
end

ex = E`JuliaRL_PPO_Nature`
run(ex)
serialize("policies/$trial_id.jls", ex.policy)
# step_through_env(ex.env, policy)
