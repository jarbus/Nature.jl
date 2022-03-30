using Nature
using Serialization

p = deserialize("/home/jack/.julia/dev/Nature/policies/2022-03-30 10:52:35 clip=0.1 uf=64 maxsteps=1000000 ws=(64, 64) nf=2.jls")

env = NatureEnv(num_starting_players=36,
                world_size=(64, 64, 4),
                window = 3,
                food_generators=[
                    FoodGen([15,15],[60,60]),
                    FoodGen([45,45],[60,60]),
                   ])
step_through_env(env, p)
