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

MAX_STEPS=200_000

N_STARTING_PLAYERS = 36
UPDATE_FREQ = 128
WORLD_SIZE = (64, 64)
CLIP = 0.1f0

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
                    max_step=500,
                    window=3,
                    food_generators=[
                        FoodGen([15,15],[60,60]),
                        FoodGen([45,45],[60,60]),
                       ])
    RLBase.reset!(env)


    global trial_id = "$timestamp clip=$CLIP uf=$UPDATE_FREQ maxsteps=$MAX_STEPS ws=$(WORLD_SIZE[1:2]) nf=$(length(env.food_generators))"

    runs = readdir("tensorboard_logs/")
    for (i, run) in enumerate(runs)
        println(i, ". ", run)
    end
    println("n. (NEW RUN)")
    print("Select a run: ")
    trial = readline()
    if trial == "n"
        println("Default name: $trial_id")
        print(  "Enter new name (empty for default) ")
        name = readline()
        if name != ""
            global trial_id = name
        end
        agents = false
        println("Creating new run: $trial_id")
    else
        global trial_id = runs[parse(Int, trial)]
        println("Resuming run: $trial_id")
        agents = deserialize("policies/$trial_id.jls")
    end

    print("Enable TensorBoard Logging? y/n: ")
    if readline() == "y"
        disable_logging(Logging.Debug)
        lg=TBLogger("tensorboard_logs/$trial_id", tb_append)
        global_logger(lg)
    else
        disable_logging(Logging.Info)
        global_logger(ConsoleLogger())
    end

    # We don't build agents at agents = false because
    # we don't want to compile between user inputs
    if agents == false
        agents = build_MultiPPOManager(env, N_STARTING_PLAYERS, UPDATE_FREQ, CLIP)
    end

    stop_condition = StopAfterStep(MAX_STEPS, is_show_progress=!haskey(ENV, "CI"))
    hook = NatureHook(env)

    Experiment(agents, env, stop_condition, hook, "# PPO with Nature")
end

ex = E`JuliaRL_PPO_Nature`
run(ex)
serialize("policies/$trial_id.jls", ex.policy)
# step_through_env(ex.env, policy)
