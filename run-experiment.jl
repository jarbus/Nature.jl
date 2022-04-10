using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using Nature
using ArgParse
# using Serialization
using TensorBoardLogger, Logging
using Infiltrator
using Dates
using Serialization

N_STARTING_PLAYERS = 36
UPDATE_FREQ = 128
WORLD_SIZE = (64, 64)
CLIP = 0.1f0
seed = 123

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--episode-len"
            help = "Maximum number of steps to run"
            arg_type = Int
            default = 500
        "--max-steps"
            help = "Maximum number of steps to run"
            arg_type = Int
            default = 1_000_000
        "--resume"
            help = "an option without argument, i.e. a flag"
            action = :store_true
        "--tb"
            help = "Log to Tensorboard"
            action = :store_true
        "name"
            help = "Name of Run"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

args = parse_commandline()
trial_id = args["name"]

rng = StableRNG(seed)
env = NatureEnv(num_starting_players=N_STARTING_PLAYERS,
                world_size=WORLD_SIZE,
                episode_len=args["episode-len"],
                window=3,
                num_frames=4,
                vocab_size=5,
                food_generators=[
                    FoodGen([15,15],[60,60]),
                    FoodGen([45,45],[60,60]),
                   ])
RLBase.reset!(env)


if args["tb"]
    println("Tensorboard Logging enabled.")
    disable_logging(Logging.Debug)
    lg=TBLogger("tensorboard_logs/$trial_id", tb_append)
    global_logger(lg)
else
    println("Tensorboard Logging not enabled.")
    disable_logging(Logging.Info)
    global_logger(ConsoleLogger())
end

if !args["resume"]
    println("Creating new run $trial_id")
    agents = build_MultiPPOManager(env, N_STARTING_PLAYERS, UPDATE_FREQ, CLIP)
else
    println("Resuming $trial_id")
    agents = deserialize("policies/$trial_id.jls")
    println(agents)
end

stop_condition = StopAfterStep(args["max-steps"], is_show_progress=!haskey(ENV, "CI"))
hook = NatureHook(env, trial_id, args["max-steps"])

ex = Experiment(agents, env, stop_condition, hook, "# PPO with Nature")

run(ex)
serialize("policies/$trial_id.jls", ex.policy)
