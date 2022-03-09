mutable struct Player{F}
    pos::Tuple{Int, Int}
    food_counts::NTuple{F, Int}
    dead::Bool
end

struct FoodGen
	smallest_pos::Tuple{Int, Int}
	biggest_pos::Tuple{Int, Int}
	dist::Distribution
end

mutable struct NatureEnv{F} <: AbstractEnv
    # Benchmark bitarray, figure out whether they should be structs or mutable
    step::Int
    num_starting_players::Int
    players::Vector{Player{F}}
    food_types::Int
    food_frames::Vector{SparseMatrixCSC{Int, Int}}
    food_generators::Vector{FoodGen}
    observation_size::NTuple{3, Int}
    observation_space::Space{Array{ClosedInterval{Int}, 3}}
end
