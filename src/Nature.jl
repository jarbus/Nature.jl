# Write tests
# Batch normalization may be needed once more food is collected and traded
# TODO Single agent collects food
module Nature

using ReinforcementLearning
using IntervalSets
using SparseArrays
using Distributions
using LinearAlgebra
using StableRNGs
using Random
using Flux
using Debugger
import Base: size
import ReinforcementLearning.RLZoo: EnrichedAction

export NatureEnv, FoodGen, reset!, remove!
include("./structs.jl")
include("./utils.jl")
include("./act.jl")
include("./player.jl")
include("./food.jl")
include("./natureenv.jl")
include("./multippo.jl")
include("./debug.jl")
# Write your package code here.

end
