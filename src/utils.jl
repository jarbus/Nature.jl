vint = Vector{Int}
function make_frame(width::Int, height::Int, window::Int)
    frame = fill(-1f0, width+(2*window), height+(2*window))
    frame[window+1:end-window,window+1:end-window] .= 0
    frame
end

small_dd_builder() = DefaultDict{Int, Float32}(0f0)
big_dd_builder() = DefaultDict{Int, DefaultDict{Int, Float32}}(()->small_dd_builder())
bigger_dd_builder() = DefaultDict{NTuple{2, Float32}, DefaultDict{Int, DefaultDict{Int, Float32}}}(()->big_dd_builder())

rand_pos(env::NatureEnv) = rand(2) .* size(env) .|> x->ceil(Int,x)
outofbounds(env::NatureEnv, pos::Tuple{Int, Int}) = !all((1,1) .<= pos .<= size(env))

($)(a::Real, b::Real) = a:sign(b):a+b-sign(b)


function onehot(idx::Int, len::Int)
    z = zeros(len)
    z[idx] = 1
    z
end

function ecat(args...)
    cat(args..., dims=ndims(args[1])+1)
end


# Clips an n-dimensional vec/tuple between bounds at each dim
function clip(x::Union{Tuple,Vector},mins::Tuple,maxes::Tuple)
    Tuple(clamp(args...) for args in zip(x, mins, maxes))
end
