export MultiPPOManager

mutable struct MultiPPOManager <: AbstractPolicy
    agents::Dict{Int, Agent}
    agents_inv::Dict{Agent, Vector{Int}}
    traces
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
end


function update_trajectory!(
    agent::Agent,
    env::AbstractEnv,
    ::PostActStage,
    players::Vector{Int}
)
    reward_dict = reward(env, players)
    term_dict = is_terminated(env, players)
    push!(agent.trajectory[:reward], ecat([reward_dict[p] for p in players]...))
    push!(agent.trajectory[:terminal], ecat([term_dict[p]   for p in players]...))
end
