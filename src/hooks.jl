mutable struct NatureHook10 <: AbstractHook
    step::Int
    max_steps::Int
    food_counts::Vector{Float32}
    act_counts::Vector{Int}
    act_probs::Vector{Float32}
    player_acts::Vector{Vector{Int}}
    total_rewards::Vector{Float32}
    trial_id::String
end
function NatureHook10(env::NatureEnv, trial_id::String, max_steps::Int)
    NatureHook(0,
        max_steps,
        zeros(Float32, env.food_types),
        zeros(length(action_space(env, 1))),
        Vector{Float32}(),
        [zeros(length(action_space(env, 1))) for i in 1:length(env.players)],
        zeros(length(env.players)),
        trial_id,
       )
end
NatureHook = NatureHook10

function (hook::NatureHook)(::PreActStage, policy, env, action)
    for p in 1:length(env.players)
        hook.act_counts[action[p].action] += 1
        push!(hook.act_probs, exp(action[p].meta.action_log_prob[1]))
        hook.player_acts[p][action[p].action] += 1
    end
end

function (hook::NatureHook)(::PostActStage, policy, env)
    for p in 1:length(env.players)
        hook.total_rewards[p] += reward(env, p)
        hook.food_counts = hook.food_counts .+ env.players[p].food_counts
    end
    hook.step += 1
    steps_per_checkpoint = 1_000
    if hook.step % steps_per_checkpoint == 0
        log_step = if hasproperty(global_logger(), :global_step)
            global_logger().global_step
        else -1 end
        serialize("checkpoints/$(hook.trial_id)/$(Int(hook.step / steps_per_checkpoint)).jls",
                  Dict(:policy => policy,
                       :tb_step => log_step,
                       :hook_step => hook.step))
    end

end

function (hook::NatureHook)(::PostEpisodeStage, policy, env)

    @info "episode"     len=env.step                       log_step_increment=0
    @info "total_food"  total_food=Tuple(hook.food_counts) log_step_increment=0
    @info "actions"     act=Tuple(hook.act_counts)         log_step_increment=0
    @info "rewards"     reward=Tuple(hook.total_rewards)   log_step_increment=0
    @info "rewards"     total_reward=sum(hook.total_rewards)
    @info "act_prob"    act_prob=mean(hook.act_probs)      log_step_increment=0
    @info "player_acts" player_acts=Tuple([Tuple(hook.player_acts[p]) for p in 1:length(env.players)]) log_step_increment=0
    @info "exchanges" ex=Tuple(env.exchanges)


    hook.act_counts    = zeros(length(action_space(env, 1)))
    hook.total_rewards = zeros(length(env.players))
    hook.food_counts   = zeros(env.food_types)
    hook.act_probs      = Vector{Float32}()

    hook.player_acts = [zeros(length(action_space(env, 1))) for _ in env.players]
end

