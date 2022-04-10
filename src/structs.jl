mutable struct Player1{F}
    pos::Tuple{Int, Int}
    food_counts::NTuple{F, Float32}
    dead::Bool
    starting_food::Int
end
Player = Player1

struct FoodGen1
	smallest_pos::Tuple{Int, Int}
	biggest_pos::Tuple{Int, Int}
	dist::Distribution
end
FoodGen = FoodGen1

mutable struct NatureEnv12{F} <: AbstractEnv
    # Benchmark bitarray, figure out whether they should be structs or mutable
    step::Int
    episode_len::Int
    num_starting_players::Int
    players::Vector{Player{F}}
    food_types::Int
    food_frames::Vector{Matrix{Float32}}
    food_generators::Vector{FoodGen}
    player_starting_food::Int
    world_size::NTuple{2, Int}
    window::Int
    obs_size::NTuple{3, Int}
    observation_space::Space{Array{ClosedInterval{Int}, 3}}
    place_record::DefaultDict
    exchanges::Vector{Float32}
    vocab_size::Int
    comms::Vector{Vector{Tuple{Int, Int, Int, Int}}}
    num_frames::Int
    past_states::Vector{Vector{Array{Float32, 3}}}
end
NatureEnv = NatureEnv12
