export MultiPPOManager

mutable struct MultiPPOManager <: AbstractPolicy
    agents::Dict{Int, Agent}
    agents_inv::Dict{Agent, Vector{Int}}
    traces
    batch_size::Int
    rng::AbstractRNG
end

Base.getindex(A::MultiPPOManager, x) = getindex(A.agents, x)
# used for simultaneous environments.
function (π::MultiPPOManager)(env::AbstractEnv)
    actions = Dict()
    for (agent::Agent, players) in π.agents_inv

        states = [state(env, p) for p in players]
        states = cat(states..., dims=ndims(states[1])+1)
        # no mask rn
        dists = prob(agent.policy, states, nothing)

        for (p, dist) in zip(players, dists)
            action = rand.(agent.policy.rng, dist)
            if ndims(action) == 2
                action_log_prob = sum(logpdf.(dist, action), dims = 1)
            else
                action_log_prob = logpdf.(dist, action)
            end

            actions[p] = EnrichedAction(action; action_log_prob = [action_log_prob])
        end
    end
    actions
end


function (π::MultiPPOManager)(stage::PreActStage, env::AbstractEnv, actions)
    # update each agent's trajectory.
    for (agent, players) in π.agents_inv
        update_trajectory!(agent, env, stage, actions, players)
        RLBase.update!(agent.policy, agent.trajectory, env, stage)
    end
end


function (π::MultiPPOManager)(stage::PostActStage, env::AbstractEnv)
    # only need to update trajectory.
    for (agent, players) in π.agents_inv
        update_trajectory!(agent, env, stage, players)
    end
end

function (π::MultiPPOManager)(stage::PreEpisodeStage, env::AbstractEnv)
    # collect state and a dummy action to each agent's trajectory here.
    for (agent, players) in π.agents_inv
        update_trajectory!(agent, env, stage, players)
    end
end


function (π::MultiPPOManager)(stage::PostEpisodeStage, env::AbstractEnv)
    # collect state and a dummy action to each agent's trajectory here.
    for (agent, players) in π.agents_inv
        update_trajectory!(agent, env, stage, players)
    end
end

function update_trajectory!(
    agent::Agent,
    env::AbstractEnv,
    ::PreEpisodeStage,
    players::Vector{Int}
)
    t = agent.trajectory
    if length(t) > 0
        pop!(t[:state])
        pop!(t[:action])
        pop!(t[:action_log_prob])
    end
end

function update_trajectory!(
    agent::Agent,
    env::AbstractEnv,
    ::PostEpisodeStage,
    players::Vector{Int}
)
    # Note that for trajectories like `CircularArraySARTTrajectory`, data are
    # stored in a SARSA format, which means we still need to generate a dummy
    # action at the end of an episode.

    state_dict      = state(env, players)
    states          = ecat([state_dict[p] for p in players]...)
    actions         = [RLCore.get_dummy_action(action_space(env, p)) for p in players]
    action_log_prob = [0.0 for p in players]

    push!(agent.trajectory;
          state           = states,
          action          = actions,
          action_log_prob = action_log_prob,)

    # if haskey(trajectory, :legal_actions_mask)
    #     lasm =
    #         policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
    #         legal_action_space_mask(env)
    #     push!(trajectory[:legal_actions_mask], lasm)
    # end
end


function update_trajectory!(
    agent::Agent,
    env::AbstractEnv,
    ::PreActStage,
    actions::Dict,
    players::Vector{Int}
)
    state_dict = state(env, players)

    push!(agent.trajectory;
          state           = ecat([state_dict[p] for p in players]...),
          action          = ecat([actions[p].action for p in players]...),
          action_log_prob = ecat([actions[p].meta.action_log_prob[1] for p in players]...))
    # if haskey(agent.trajectory, :legal_actions_mask)
    #     lasm =
    #         policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
    #         legal_action_space_mask(env)
    #     push!(agent.trajectory[:legal_actions_mask], lasm)
    # end
end


function update_trajectory!(
    agent::Agent,
    env::AbstractEnv,
    ::PostActStage,
    players::Vector{Int}
)

    reward_dict = reward(env, players)
    t = agent.trajectory
    term_dict = is_terminated(env, players)

    # println(agent.policy.update_step)
    # println(t[:action][:,1:length(t)][t[:reward] .> 0])
    # println(length(t))
    # println(size(t[:action]))
    # @infiltrate !(unique(t[:action][:,1:length(t)][t[:reward] .> 0]) in ([],[5]))
    push!(agent.trajectory[:reward], ecat([reward_dict[p] for p in players]...))
    push!(agent.trajectory[:terminal], ecat([term_dict[p]   for p in players]...))
    # println(agent.trajectory[:action][:,end])
    # println(agent.trajectory[:reward][:,end])
end

# function RLBase.update!(
#     trajectory::AbstractTrajectory,
#     policy::AbstractPolicy,
#     env::AbstractEnv,
#     ::PostEpisodeStage,
# )
#     # Note that for trajectories like `CircularArraySARTTrajectory`, data are
#     # stored in a SARSA format, which means we still need to generate a dummy
#     # action at the end of an episode.

#     s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)

#     action_space = policy isa NamedPolicy ? action_space(env, nameof(policy)) : action_space(env)
#     a = get_dummy_action(action_space)

#     push!(trajectory[:state], s)
#     push!(trajectory[:action], a)
#     if haskey(trajectory, :legal_actions_mask)
#         lasm =
#             policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
#             legal_action_space_mask(env)
#         push!(trajectory[:legal_actions_mask], lasm)
#     end
# end

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

# function RLBase.prob(p::PPOPolicy, env::NatureEnv, player::Int)
#     s = state(env, player)
#     s = Flux.unsqueeze(s, ndims(s) + 1)
#     mask =  ActionStyle(env) === FULL_ACTION_SET ? legal_action_space_mask(env) : nothing
#     prob(p, s, mask)
# end

# function (p::PPOPolicy)(env::AbstractEnv, player::Int)
#     dist =  prob(p, env, player)
#     action = rand.(p.rng,dist)
#     if ndims(action) == 2
#         action_log_prob = sum(logpdf.(dist, action), dims = 1)
#     else
#         action_log_prob = logpdf.(dist, action)
#     end
#     a = EnrichedAction(action; action_log_prob = vec(action_log_prob))
#     a
# end
