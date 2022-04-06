function Player1(x::Int,y::Int, num_food_types::Int, starting_food::Int)
    # Yes these types are necessary
    Player{num_food_types}(
        (x,y),
        NTuple{num_food_types, Int}(starting_food for i in 1:num_food_types),
        false,
        starting_food)
end
