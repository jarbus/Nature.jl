export MultiPPOManager

"""
    MultiPPOManager(; agents::Dict{<:Any, <:Agent}, args...)
Multi-agent Deep Deterministic Policy Gradient(MultiPPO) implemented in Julia. By default, `MultiPPOManager` uses for simultaneous environments with [continuous action space](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#stochastic-policies).
See the paper https://arxiv.org/abs/1706.02275 for more details.

# Keyword arguments
- `agents::Dict{<:Any, <:Agent}`, here each agent collects its own information. While updating the policy, each **critic** will assemble all agents'
  trajectory to update its own network. **Note that** here the policy of the `Agent` should be `DDPGPolicy` wrapped by `NamedPolicy`, see the relative
  experiment([`MultiPPO_KuhnPoker`](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MultiPPO_KuhnPoker/#JuliaRL\\_MultiPPO\\_KuhnPoker) or [`MultiPPO_SpeakerListener`](https://juliareinforcementlearning.org/docs/experiments/experiments/Policy%20Gradient/JuliaRL_MultiPPO_SpeakerListener/#JuliaRL\\_MultiPPO\\_SpeakerListener)) for references.
- `traces`, set to `SARTS` if you are apply to an environment of `MINIMAL_ACTION_SET`, or `SLARTSL` if you are to apply to an environment of `FULL_ACTION_SET`.
- `batch_size::Int`
- `update_freq::Int`
- `update_step::Int`, count the step.
- `rng::AbstractRNG`.
"""
mutable struct MultiPPOManager <: AbstractPolicy
    agents::Dict{Int, Agent}
    traces
    batch_size::Int
    update_freq::Int
    update_step::Int
    rng::AbstractRNG
end

Base.getindex(A::MultiPPOManager, x) = getindex(A.agents, x)
# used for simultaneous environments.
function (π::MultiPPOManager)(env::AbstractEnv)
    Dict(player => agent.policy(env, player)
    for (player, agent) in π.agents)
end


function (p::PPOPolicy)(env::AbstractEnv, player::Int)
    dist =  prob(p, env, player)
    action = rand.(p.rng,dist)
    if ndims(action) == 2
        action_log_prob = sum(logpdf.(dist, action), dims = 1)
    else
        action_log_prob = logpdf.(dist, action)
    end
    a = EnrichedAction(action; action_log_prob = vec(action_log_prob))
    # println(a)
    a
end


function RLBase.prob(p::PPOPolicy, env::AbstractEnv, player::Int)
    s = state(env, player)
    s = Flux.unsqueeze(s, ndims(s) + 1)
    mask =  ActionStyle(env) === FULL_ACTION_SET ? legal_action_space_mask(env) : nothing
    prob(p, s, mask)
end

#RLBase.prob(A::MultiAgentManager, env::AbstractEnv, args...) = prob(A[current_player(env)].policy, env, args...)

function (π::MultiPPOManager)(stage::PostActStage, env::AbstractEnv)
    # only need to update trajectory.
    for (p, agent) in π.agents
        update!(agent.trajectory, agent.policy, env, stage, p)
    end
end

function (π::MultiPPOManager)(stage::PreActStage, env::AbstractEnv, actions)
    # update each agent's trajectory.
    for (player, agent) in π.agents

        update!(agent.trajectory, agent.policy, env, stage, player, actions[player])
        update!(agent.policy, agent.trajectory, env, stage)

    end

    # update policy
    # update!(π.agents[1].policy, env)
end

# function (π::MultiPPOManager)(stage::PostEpisodeStage, env::AbstractEnv)
#     # collect state and a dummy action to each agent's trajectory here.
#     for (_, agent) in π.agents
#         update!(agent.trajectory, agent.policy, env, stage)
#     end

#     # update policy
#     update!(π, env)
# end


function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PreActStage,
    player::Int,
    action::EnrichedAction,
)
    @bp
    s = policy isa NamedPolicy ? state(env, player, nameof(policy)) : state(env, player)

    push!(trajectory;
            state=s,
            action=action.action,
            action_log_prob=action.meta.action_log_prob)
    if haskey(trajectory, :legal_actions_mask)
        lasm =
            policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
            legal_action_space_mask(env)
        push!(trajectory[:legal_actions_mask], lasm)
    end
end


function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::AbstractPolicy,
    env::AbstractEnv,
    ::PostActStage,
    player::Int,
)
    r = policy isa NamedPolicy ? reward(env, player) : reward(env, player)
    push!(trajectory[:reward], r)
    push!(trajectory[:terminal], is_terminated(env))
end


# function RLBase.update!(
#     p::PPOPolicy,
#     t::Union{PPOTrajectory,MaskedPPOTrajectory},
#     ::AbstractEnv,
#     ::PreActStage,
# )
#     length(t) == 0 && return  # in the first update, only state & action are inserted into trajectory
#     p.update_step += 1
#     if p.update_step % p.update_freq == 0
#         RLBase._update!(p, t)
#     end
# end
# function RLBase.update!(
#     trajectory::Union{PPOTrajectory,MaskedPPOTrajectory},
#     ::PPOPolicy,
#     env::MultiThreadEnv,
#     ::PreActStage,
#     player::Int,
#     action::EnrichedAction,
# )
#     push!(
#         trajectory;
#         state = state(env, player),
#         action = action.action,
#         action_log_prob = action.meta.action_log_prob,
#     )

#     if trajectory isa MaskedPPOTrajectory
#         push!(trajectory; legal_actions_mask = legal_action_space_mask(env))
#     end
# end
