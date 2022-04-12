using Nature
using Serialization

pols = readdir("checkpoints/")
for (i, pol) in enumerate(pols)
    println(i, ". ", pol)
end
print("Select a policy: ")
pol_name = "checkpoints/"*pols[parse(Int, readline())]

p = deserialize(pol_name)

env = NatureEnv(num_starting_players=36,
                world_size=(64, 64),
                window = 3,
                episode_len=500,
                vocab_size=5,
                food_generators=[
                    FoodGen([15,15],[60,60]),
                    FoodGen([45,45],[60,60]),
                   ])
step_through_env(env, p)
