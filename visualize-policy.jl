using Nature
using Serialization

p = deserialize(readdir("policies/", join=true)[end])

env = NatureEnv(num_starting_players=36,
                world_size=(64, 64),
                window = 3,
                food_generators=[
                    FoodGen([15,15],[60,60]),
                    FoodGen([45,45],[60,60]),
                   ])
step_through_env(env, p)
