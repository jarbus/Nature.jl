# Directions
const directions = [(-1, 0),(1, 0),(0, 1),(0, -1)]
# For readability
PICK_FOOD_1 = 5
PLACE_FOOD_1 = 6
# Perform a move action
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
    place_amount = 0.5

    if pick && num_food > 0
        player.food_counts = player.food_counts .+ Tuple(num_food * onehot(food_type, env.food_types))
        for (placed_player, placed_count) in env.place_record[player.pos][food_type]
            if placed_player != p
                env.exchanges[food_type] += placed_count
            end
        end
        fframe[player.pos...] = 0
        env.place_record[player.pos][food_type] = small_dd_builder()    elseif place && player.food_counts[food_type] >= place_amount
        player.food_counts = Tuple(player.food_counts .-  (place_amount .* onehot(food_type, env.food_types)))
        fframe[player.pos...] += place_amount
        # You need this
        f_::Float32 = env.place_record[player.pos][food_type][p]
        env.place_record[player.pos][food_type][p] = f_ + place_amount
    end
end

function comm(env::NatureEnv, p::Int, symbol::Int)
end
