# function visualize(env::NatureEnv)
#     unicodeplots()

#     scatter()
#     food_vecs = [Tuple.(indices(sp)) for sp in env.food_frames]
#     food_colors = [:orange for _ in 1:env.food_types]
#     agent_colors = [:red for _ in 1:length(env.players)]
#     for (ft, (fv, color)) in enumerate(zip(food_vecs, food_colors))
#         scatter!(fv,c=color, label="f$ft")
#     end
#     for (p, color) in zip(keys(env.players), agent_colors)
#         scatter!(env.players[p].pos,c=color, label="a$p", markershape=:square)
#     end
#     scatter!(xlim=(1,32), ylim=(1,32))
# end

function step_through_env(env::NatureEnv, policy::T) where {T <: MultiPPOManager}
    reset!(env)

    get_player_poses() = [p.pos for p in env.players]
    get_food_poses() = [[(x, y)
                        for (x, y, _) in zip(findnz(env.food_frames[f])...)]
                        for f in env.food_types]

    player_poses = []
    food_poses = []
    food_colors = (:green, :orange, :blue, :red)
    nframes = 80
    frame = 0

    while !is_terminated(env) && frame < nframes
        env |> policy |> env
        push!(player_poses, get_player_poses())
        push!(food_poses, get_food_poses())
        frame += 1
    end

    fig = Figure()
    ax  = Axis(fig[1,1])
    limits!(ax, 0, env.observation_size[1], 0, env.observation_size[2])
    sl_t = Slider(fig[2, 1], range = 1:1:frame, startvalue = 1)

    pps = @lift player_poses[$(sl_t.value)]
    GLMakie.scatter!(pps, color=:pink, markersize=60)
    for f in 1:env.food_types
        fps = @lift food_poses[$(sl_t.value)][f]
        GLMakie.scatter!(fps, color=food_colors[f], markersize=40)
    end

    display(fig)
    nothing
end




