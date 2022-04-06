# NOTE: MAKE SURE PLAYER HAS ENOUGH OF EACH TYPE OF FOOD
# player p moves to and collects food type ft
function move_and_collect(env::NatureEnv, p::Int, ft::Int)
    fframe = env.food_frames[ft]
    foods = Tuple.(filter(pos->fframe[pos]>0f0, CartesianIndices(fframe)))

    for food in foods
        while true
            if env.players[p].dead
                return
            end
            dir = food .- env.players[p].pos
            dir = dir ./ abs.(dir) |> i->map(y->isnan(y) ? 0 : y,i)
            # Randomly mask x or y if
            # direction is diagonal
            if abs.(dir) == (1.,1.)
                mask = zeros(2)
                mask[rand() |> round |> x->x+1 |> Int] = 1
                dir = Tuple(dir .* mask)
            end

            dir = Int.(dir)
            action = findfirst(x->x==dir, directions)
            # action might be (0, 0), which returns nothing in this case
            if !isnothing(action)
                env(action, p)
            end
            env(4 + (2 * (ft-1))+1, p) # eat food
            env.players[p].pos == food && break
        end
    end
end

function visualize_food(env::NatureEnv)
    sparse(sum(env.food_frames))
end
