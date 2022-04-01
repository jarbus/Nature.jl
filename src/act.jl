# Directions

# Perform a move action
const directions = [(-1, 0),(1, 0),(0, 1),(0, -1)]
function move(env::NatureEnv, player::Int, dir::Int)
    new_pos = env.players[player].pos .+ directions[dir]
    outofbounds(env, new_pos) && return
    env.players[player].pos = new_pos
end
# Perform a food action
function food(env::NatureEnv, p::Int, idx::Int)
    # [ place1, pick1, place2, pick2, ... placef, pickf]
    pick  = idx % 2 == 1
    place = idx % 2 == 0
    food_type = floor(Int, ((idx - 1) / 2)) + 1
    fframe   = env.food_frames[food_type]
    player   = env.players[p]
    num_food = fframe[player.pos...]
    place_amount = 0.1

    if pick && num_food > 0
        player.food_counts = player.food_counts .+ Tuple(num_food * onehot(food_type, env.food_types))
        fframe[player.pos...] = 0
    elseif place && player.food_counts[food_type] >= place_amount
        player.food_counts = Tuple(player.food_counts .-  (place_amount .* onehot(food_type, env.food_types)))
        fframe[player.pos...] += place_amount
    end
end
