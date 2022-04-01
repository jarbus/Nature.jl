function Player1(x::Int,y::Int, num_food_types::Int)
    # Yes these types are necessary
    Player{num_food_types}(
        (x,y),
        NTuple{num_food_types, Int}(3f0 for i in 1:num_food_types),
        false)
end
