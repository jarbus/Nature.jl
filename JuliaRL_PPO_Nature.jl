using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Nature
using Plots
using Debugger


mutable struct FoodCountHook1 <: AbstractHook
    step::Int
    food_counts::Dict{Int, Dict{Int, Vector{Int}}}
end
function FoodCountHook1(env::NatureEnv)
    FoodCountHook(0,
        Dict(
            i=>Dict(f=>[] for f in 1:env.food_types)
            for i in 1:length(env.players)))
end
FoodCountHook = FoodCountHook1


function (hook::FoodCountHook)(::PostActStage, policy, env)
    for (p, player) in enumerate(env.players)
        for f in 1:env.food_types
            push!(hook.food_counts[p][f], player.food_counts[f])
        end
    end
end


function RL.Experiment(
    ::Val{:JuliaRL},
    ::Val{:PPO},
    ::Val{:Nature},
    ::Nothing;
    save_dir = nothing,
    seed = 123,
)
    rng = StableRNG(seed)
    N_STARTING_PLAYERS = 8
    N_STARTING_PLAYERS=2
    UPDATE_FREQ = 32
    env = NatureEnv(num_starting_players=N_STARTING_PLAYERS)
    reset!(env)
    ns, na = size(state(env, 1)), length(action_space(env,1))
    println(ns, na)

    create_actor() = Chain(
        Conv((3,3), 4=>16, relu),
        Flux.flatten,
        Dense(14400, 64, relu; init = glorot_uniform(rng)),
        Dense(64, na; init = glorot_uniform(rng)),
    )

    create_critic() = Chain(
        Conv((3,3),4=>16, relu),
        Flux.flatten,
        Dense(14400, 64, relu; init = glorot_uniform(rng)),
        Dense(64, 1; init = glorot_uniform(rng)),
    )

    create_trajectory() = PPOTrajectory(;
        capacity = UPDATE_FREQ,
        state = Array{Float32, 4} => (ns..., 1),
        action = Vector{Int} => (1,),
        action_log_prob = Vector{Float32} => (1,),
        reward = Vector{Float32} => (1,),
        terminal = Vector{Bool} => (1,),
    )

    # TODO CHANGE THIS TO BE DYNAMIC UPDATE FREQ AS NUM AGENTS CHANGES
    function create_policy()
        PPOPolicy(
            approximator = ActorCritic(
                actor = create_actor(),
                critic = create_critic(),
                optimizer = ADAM(1e-3),
            ),
            γ = 0.99f0,
            λ = 0.95f0,
            clip_range = 0.1f0,
            max_grad_norm = 0.5f0,
            n_epochs = 4,
            n_microbatches = 4,
            actor_loss_weight = 1.0f0,
            critic_loss_weight = 0.5f0,
            entropy_loss_weight = 0.001f0,
            update_freq = UPDATE_FREQ*N_STARTING_PLAYERS,
        )
    end
    shared_policy = create_policy()

    create_agent() = Agent(
        policy = shared_policy,
        trajectory = create_trajectory()
    )

    t = create_trajectory()

    agents = MultiPPOManager(
        Dict(player => create_agent() for player in 1:N_STARTING_PLAYERS),
        PPOTrajectory, # trace's type
        512, # batch_size
        100, # update_freq
        0, # initial update_step
        rng
    )

    stop_condition = StopAfterStep(10_000, is_show_progress=!haskey(ENV, "CI"))
    hook = FoodCountHook(env)
    Experiment(agents, env, stop_condition, hook, "# PPO with Nature")

end

# using Plots
# using Statistics
# # pyplot() #hide
plot();
ex = E`JuliaRL_PPO_Nature`
run(ex)
for p in 1:length(ex.env.players)
    for f in 1:ex.env.food_types
        plot!(ex.hook.food_counts[p][f], label="player $p food $f")
    end
end
# n = minimum(map(length, ex.hook.rewards))
# m = mean([@view(x[1:n]) for x in ex.hook.rewards])
# s = std([@view(x[1:n]) for x in ex.hook.rewards])
# plot(m,ribbon=s)
# savefig("assets/JuliaRL_PPO_Nature.png") #hide
