# Write tests
# Batch normalization may be needed once more food is collected and traded
module Nature

using ReinforcementLearning
using IntervalSets
using SparseArrays
using Distributions
using LinearAlgebra
using StableRNGs
using Random
using Flux
using Infiltrator
using GLMakie
import Base: size
import ReinforcementLearning.RLZoo: EnrichedAction

export NatureEnv, FoodGen, reset!, remove!, visualize, step_through_env, MultiPPOManager, NatureHook, ($), visualize_food
include("./structs.jl")
include("./utils.jl")
include("./act.jl")
include("./player.jl")
include("./food.jl")
include("./natureenv.jl")
include("./multippo.jl")
include("./debug.jl")
include("./visualize.jl")
include("./hooks.jl")
# Write your package code here.

end
