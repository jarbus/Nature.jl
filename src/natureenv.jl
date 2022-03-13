# SparseArrays.dropstored!
size(env::NatureEnv) = env.observation_size[1:2]

function NatureEnv(;
        num_starting_players=2,
        observation_size=(32, 32, 4),
        food_generators=[
            FoodGen([5,5],[1,1]),
            FoodGen([25,25],[1,1]),
           ]) #width, heigh, channels, batch

    num_food_types = length(food_generators)

    observation_space = Space(
        ClosedInterval.(
            fill(typemin(Int), observation_size),
            fill(typemax(Int), observation_size)
        )
    )

    NatureEnv{num_food_types}(
        0,
        num_starting_players,
        [],
        num_food_types,
        Vector{SparseMatrixCSC{Int, Int}}(),
        food_generators,
        observation_size,
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
RLBase.reward(env::NatureEnv, player::Int) = sum(fc for fc in env.players[player].food_counts)

RLBase.is_terminated(env::NatureEnv, player::Int) = env.players[player].dead || env.step == 100
RLBase.is_terminated(env::NatureEnv, players::Vector{Int}) = Dict(p=>is_terminated(env, p) for p in players)
function RLBase.is_terminated(env::NatureEnv)
    if all(is_terminated(env, p) for p in keys(env.players)) ||
        (unique(sum(env.food_frames))) == [0] ||
        env.step == 100
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
        push!(env.food_frames, make_frame(size(env)...))
        for nf::Tuple in new_food(env.food_generators[type], 100)
            env.food_frames[type][nf...] += 1
        end
    end

    env.players = [Player(rand_pos(env)...,env.food_types) for _ in 1:env.num_starting_players]
end

function sparse_state(env::NatureEnv, player::Int)
    self_frame = make_frame(size(env)...)
    other_frame = make_frame(size(env)...)
    for i in 1:length(env.players)
        if i == player
            self_frame[env.players[i].pos...] = 1
        else
            other_frame[env.players[i].pos...] = 1
        end
    end
    [self_frame, other_frame, env.food_frames...]
end

function RLBase.state(env::NatureEnv, player::Int)
    frames = sparse_state(env, player) .|> Matrix{Float32}
    frames = cat(frames..., dims=3)
    frames
end

function (env::NatureEnv)(actions::Dict)
    env.step += 1
    Dict(p=>env(a.action[1], p) for (p, a) in actions)
end

function (env::NatureEnv)(action::Int, player::Int)

    env.players[player].food_counts = env.players[player].food_counts .* 0

    MOVE_RANGE = 1:4
    FOOD_RANGE = (MOVE_RANGE.stop+1)$(2*env.food_types)

    (action in MOVE_RANGE) && move(env, player, action)
    (action in FOOD_RANGE) && food(env, player, action - FOOD_RANGE.start + 1)
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
