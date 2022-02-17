# SparseArrays.dropstored!
mutable struct Player{F}
    pos::Tuple{Int32, Int32}
    food_counts::NTuple{F, Int32}
end

mutable struct NatureEnv{F} <: AbstractEnv
    # Benchmark bitarray, figure out whether they should be structs or mutable
    players::Vector{Player{F}}
    food_types::Int32
    food_frames::Vector{SparseMatrixCSC{Int32, Int32}}
    observation_size::NTuple{3, Int32}
    observation_space::Space{Array{ClosedInterval{Int32}, 3}}
end

Base.show(io::IO, nenv::NatureEnv) = nothing

function NatureEnv(;
        num_food_types=2,
        num_starting_players=2,
        observation_size=(64, 64, 10)) #width, heigh, channels, batch
    players = [Player{num_food_types}(
                (0,0),
                Tuple(0 for _ in 1:num_food_types))
               for __ in 1:num_starting_players]

    observation_space = Space(
        ClosedInterval.(
            fill(typemin(Int32), observation_size),
            fill(typemax(Int32), observation_size)
        )
    )

    NatureEnv{num_food_types}(
        players,
        num_food_types,
        Vector{Set{Tuple{Int32, Int32}}}(), # food_poses
        observation_size,
        observation_space
       )
end





RLBase.NumAgentStyle(env::NatureEnv) = MultiAgent(length(env.players))
RLBase.is_terminated(env::NatureEnv, player::Int32) = false
RLBase.state_space(env::NatureEnv, player::Int32) = env.observation_space
RLBase.action_space(env::NatureEnv, player::Int32) = Space()
RLBase.reward(env::NatureEnv, player::Int32) = 0
(env::NatureEnv)(action, player::Int32) = 0

function RLBase.reset!(env::NatureEnv)
    for type in 1:env.food_types
        env.food_frames = make_frame(env.observation_size[1:2]...)
    end

end


function RLBase.state(env::NatureEnv, player::Int32)
end
