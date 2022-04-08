function step_through_env(env::NatureEnv, policy::T) where {T <: MultiPPOManager}
    RLBase.reset!(env)

    get_player_poses() = [p.pos for p in env.players if !p.dead]
    function get_food_poses()
        fps = Vector()
        for f in 1:env.food_types
            push!(fps, Vector())
            for pos in CartesianIndices(env.food_frames[f])
                if env.food_frames[f][pos] > 0f0
                    push!(fps[f], (Tuple(pos)..., env.food_frames[f][pos]))
                end
            end
        end
        fps
    end

    player_poses = []
    food_poses = []
    food_color_maps = (
        (:darkblue, :lightblue),
        (:yellow, :orange))
    nframes = env.episode_len
    frame = 0

    while !is_terminated(env) && frame < nframes
        env |> policy |> env
        push!(player_poses, get_player_poses())
        push!(food_poses, get_food_poses())
        frame += 1
    end

    fig = Figure()
    ax  = Axis(fig[1,1])
    limits!(ax, 0, env.world_size[1], 0, env.world_size[2])
    lsgrid = labelslidergrid!(fig, ["step"], [1:1:frame])
    fig[2, 1] = lsgrid.layout
    slider = lsgrid.sliders[1]


    for f in 1:env.food_types
        fps = @lift [(x, y) for (x, y, _) in food_poses[$(slider.value)][f]]
        fvs = @lift [z for (_, _, z) in food_poses[$(slider.value)][f]]
        GLMakie.scatter!(fps,color=fvs, colormap=food_color_maps[f], markersize=40, marker=:rect)
    end
    pps = @lift player_poses[$(slider.value)]
    GLMakie.scatter!(pps, color=:pink, markersize=60)

    display(fig)
    nothing
end




