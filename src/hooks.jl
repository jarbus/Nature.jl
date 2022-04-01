mutable struct NatureHook10 <: AbstractHook
    step::Int
    food_counts::Vector{Float32}
    act_counts::Vector{Int}
    act_probs::Vector{Float32}
    player_acts::Vector{Vector{Int}}
    total_rewards::Vector{Float32}
end
function NatureHook10(env::NatureEnv)
    NatureHook(0,
        zeros(Float32, env.food_types),
        zeros(length(action_space(env, 1))),
        Vector{Float32}(),
        [zeros(length(action_space(env, 1))) for i in 1:length(env.players)],
        zeros(length(env.players)),
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

end


# function (hook::NatureHook)(::PreExperimentStage, policy, env)
#     @info "delta_prob" delta_prob=0                            log_step_increment=0
#     @info "advantages" adv_avg=0                               log_step_increment=1
# end

function (hook::NatureHook)(::PostEpisodeStage, policy, env)

    @info "episode"     len=env.step                       log_step_increment=0
    @info "total_food"  total_food=Tuple(hook.food_counts) log_step_increment=0
    @info "actions"     act=Tuple(hook.act_counts)         log_step_increment=0
    @info "rewards"     reward=Tuple(hook.total_rewards)   log_step_increment=0
    @info "rewards"     total_reward=sum(hook.total_rewards)
    @info "act_prob"    act_prob=mean(hook.act_probs)      log_step_increment=0
    @info "player_acts" player_acts=Tuple([Tuple(hook.player_acts[p]) for p in 1:length(env.players)])


    hook.act_counts    = zeros(length(action_space(env, 1)))
    hook.total_rewards = zeros(length(env.players))
    hook.food_counts   = zeros(env.food_types)
    hook.act_probs      = Vector{Float32}()

    hook.player_acts = [zeros(length(action_space(env, 1))) for _ in env.players]
end

