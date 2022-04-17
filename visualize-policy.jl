using Nature
using Serialization

pols = readdir("checkpoints/")
for (i, pol) in enumerate(pols)
    println(i, ". ", pol)
end
print("Select a run: ")
pol_name = "checkpoints/"*pols[parse(Int, readline())]

pols = readdir("$pol_name/")
for (i, pol) in enumerate(pols)
    println(i, ". ", pol)
end
print("Select a checkpoint: ")
pol_name = "$pol_name/"*pols[parse(Int, readline())]

p = deserialize(pol_name)[:policy]

WORLD_SIZE=(2,2)
env = NatureEnv(num_starting_players=2,
                world_size=WORLD_SIZE,
                window = 3,
                num_frames=2,
                episode_len=1000,
                vocab_size=5,
                food_generators=[
                    FoodGen([1,1],[0.0001,0.0001],max_pos=WORLD_SIZE),
                    FoodGen([2,2],[0.0001,0.0001],max_pos=WORLD_SIZE),
                   ])
step_through_env(env, p)
