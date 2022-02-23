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
    # [ pick1, place1, pick2, place2, ... pickf, placef]
    pick  = idx % 2 == 1
    place = idx % 2 == 0
    food_type = floor(Int, ((idx - 1) / 2)) + 1
    fframe   = env.food_frames[food_type]
    player   = env.players[p]
    num_food = fframe[player.pos...]

    if pick && num_food > 0
        player.food_counts = player.food_counts .+ Tuple(num_food * onehot(food_type, env.food_types))
        remove!(fframe, player.pos...)
    elseif place && player.food_counts[food_type] > 0
        player.food_counts = Tuple(player.food_counts .- onehot(food_type, env.food_types))
        fframe[player.pos...] += 1
    end
end
