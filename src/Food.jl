struct FoodGen
	smallest_pos::Tuple{Int32, Int32}
	biggest_pos::Tuple{Int32, Int32}
	dist::Distribution
end

function FoodGen(mean::Vector{<:Real},
        std::Vector{<:Real};
        min_pos=(1,1),
        max_pos=(64,64))
    FoodGen(min_pos, max_pos, MvNormal(mean, std))
end

new_food(fg::FoodGen) = clip(rand(fg.dist), fg.smallest_pos, fg.biggest_pos)

new_food(fg::FoodGen, n::Int) =
    rand(fg.dist, n) |> eachcol .|> Tuple .|>
    x->clip(x, fg.smallest_pos, fg.biggest_pos)

