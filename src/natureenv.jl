# SparseArrays.dropstored!
size(env::NatureEnv) = env.world_size[1:2]

function NatureEnv3(;
        num_starting_players=2,
        world_size=(32, 32),
        window=3,
        max_step=100,
        food_generators=[
            FoodGen([5,5],[1,1]),
            FoodGen([25,25],[1,1]),
           ]) #width, heigh, channels, batch

    num_food_types = length(food_generators)
    # self, other, x pos, y pos, food_poses, food_counts
    num_channels = 2+2+num_food_types+num_food_types

    observation_space = Space(
        ClosedInterval.(
            fill(typemin(Int), world_size..., num_channels),
            fill(typemax(Int), world_size..., num_channels)
        )
    )

    NatureEnv3{num_food_types}(
        0,
        max_step,
        num_starting_players,
        [],
        num_food_types,
        Vector{Matrix{Float32}}(),
        food_generators,
        world_size,
        window,
        (2*window+1, 2*window+1, num_channels),
        observation_space
       )
end


RLBase.state_space(env::NatureEnv,::Observation{Any}, players::Dict{Int,Player}) = Dict(p=>state_space(env, p) for p in keys(players))
RLBase.state(env::NatureEnv,::Observation{Any}, players::Vector{Int}) = Dict(p=>state(env, p) for p in players)
RLBase.action_space(env::NatureEnv, players::Dict{Int,Player}) = Dict(p=>action_space(env, p) for p in keys(players))
RLBase.reward(env::NatureEnv, players::Dict{Int,Player}) = Dict(p=>reward(env, p) for p in keys(players))
RLBase.reward(env::NatureEnv, players::Vector{Int}) = Dict(p=>reward(env, p) for p in players)

RLBase.state_space(env::NatureEnv,::Observation{Any}, player::Int) = env.observation_space
RLBase.action_space(env::NatureEnv, player::Int) = Base.OneTo(4 + 2env.food_types)

function RLBase.reward(env::NatureEnv, player::Int)
    if all(env.players[player].food_counts .> 0f0) && !env.players[player].dead
        return 1f0
    else
        return 0f0
    end
end

RLBase.is_terminated(env::NatureEnv, player::Int) = env.players[player].dead || env.step == env.max_step
RLBase.is_terminated(env::NatureEnv, players::Vector{Int}) = Dict(p=>is_terminated(env, p) for p in players)
function RLBase.is_terminated(env::NatureEnv)
    if all(is_terminated(env, p) for p in keys(env.players)) ||
        env.step == env.max_step
        return true
    else
        return false
    end
end
function RLBase.reset!(env::NatureEnv)
    # Add Food according to each generator
    env.step=0
    env.food_frames = []
    for type in 1:env.food_types
        push!(env.food_frames, zeros(Float32, size(env)...))
        for nf::Tuple in new_food(env.food_generators[type], 10000)
            env.food_frames[type][nf...] += 0.1f0
        end
    end

    env.players = [Player(rand_pos(env)...,env.food_types) for _ in 1:env.num_starting_players]
end

function RLBase.state(env::NatureEnv, player::Int)
    w = env.window
    px, py = env.players[player].pos
    self_frame = make_frame(size(env)..., w)
    other_frame = make_frame(size(env)..., w)
    if env.players[player].dead
        return zeros(Float32, env.obs_size...)
    end
    # Player frames
    for i in 1:length(env.players)
        if i == player
            self_frame[(w .+ env.players[i].pos)...] = 1
        else
            other_frame[(w .+ env.players[i].pos)...] = 1
        end
    end

    # positions
    xpos = hcat([i*ones(env.world_size[1]+(2*w)) for i in 1-w:env.world_size[1]+w]...) ./ (env.world_size[1]+w)
    ypos = vcat([i*ones(1, env.world_size[2]+(2*w)) for i in 1-w:env.world_size[2]+w]...) ./ (env.world_size[2]+w)

    # Food frames and food counts
    food_frames = []
    food_counts = []
    for (f, env_food_frame) in enumerate(env.food_frames)
        ff = make_frame(size(env)..., w)
        fc = make_frame(size(env)..., w) .* 0f0
        ff[w+1:end-w,w+1:end-w] = env_food_frame ./ 10f0
        for p in env.players
            p.dead && continue
            fc[w+p.pos[1],w+p.pos[2]] = p.food_counts[f] / 10f0
        end
        push!(food_frames, ff)
        push!(food_counts, fc)
    end
    frames = [self_frame, other_frame, xpos, ypos, food_frames..., food_counts...]
    frames = cat(frames..., dims=3)
    frames = frames[px:px+(2*w), py:py+(2*w), :]
end

function (env::NatureEnv)(actions::Dict)
    env.step += 1
    Dict(p=>env(a.action[1], p) for (p, a) in actions)
end

function (env::NatureEnv)(action::Int, player::Int)

    if max(env.players[player].food_counts...) <= 0f0
        return nothing
    end
    # Players lose 0.1 food per tick, floored at 0
    env.players[player].food_counts = env.players[player].food_counts .- 0.1 .|> x->max(x, 0)

    MOVE_RANGE = 1:4
    FOOD_RANGE = (MOVE_RANGE.stop+1)$(2*env.food_types)

    (action in MOVE_RANGE) && move(env, player, action)
    (action in FOOD_RANGE) && food(env, player, action - FOOD_RANGE.start + 1)
    if min(env.players[player].food_counts...) <= 0f0
        env.players[player].dead = true
    end
    nothing
end


RLBase.legal_action_space_mask(::NatureEnv) = nothing
RLBase.current_player(env::NatureEnv) =  Dict{Int, Player}(i=>p for (i,p) in enumerate(env.players) if !p.dead)
# https://juliareinforcementlearning.org/docs/How_to_write_a_customized_environment/#More-Complicated-Environments
RLBase.NumAgentStyle(env::NatureEnv) = MultiAgent(length(env.players))
RLBase.DynamicStyle(::NatureEnv) = SIMULTANEOUS
RLBase.ActionStyle(::NatureEnv) = FULL_ACTION_SET
RLBase.InformationStyle(::NatureEnv) = IMPERFECT_INFORMATION
# RLBase.StateStyle(::TicTacToeEnv) =
#     (Observation{Any}(), Observation{Int}(), Observation{BitArray{3}}())
# RLBase.RewardStyle(::TicTacToeEnv) = TERMINAL_REWARD
# RLBase.UtilityStyle(::TicTacToeEnv) = GENERAL_SUM
RLBase.ChanceStyle(::NatureEnv) = DETERMINISTIC
