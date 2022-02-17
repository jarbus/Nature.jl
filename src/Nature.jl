module Nature

using ReinforcementLearning
using IntervalSets
using SparseArrays
using Distributions

export NatureEnv, FoodGen
include("./utils.jl")
include("./Food.jl")
include("./NatureEnv.jl")
# Write your package code here.

end
