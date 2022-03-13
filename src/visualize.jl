function visualize(env::NatureEnv)
    unicodeplots()

    scatter()
    food_vecs = [Tuple.(indices(sp)) for sp in env.food_frames]
    food_colors = [:orange for _ in 1:env.food_types]
    agent_colors = [:red for _ in 1:length(env.players)]
    for (ft, (fv, color)) in enumerate(zip(food_vecs, food_colors))
        scatter!(fv,c=color, label="f$ft")
    end
    for (p, color) in zip(keys(env.players), agent_colors)
        scatter!(env.players[p].pos,c=color, label="a$p", markershape=:square)
    end
    scatter!(xlim=(1,32), ylim=(1,32))
end

function step_through_env(env::NatureEnv, policy::T) where {T <: MultiPPOManager}
    reset!(env)
    while !is_terminated(env)
        env |> policy |> env
        display(visualize(env))
        sleep(0.1)
    end
end
