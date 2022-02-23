vint = Vector{Int}()
make_frame(x::Int, y::Int) = sparse(vint, vint, vint, x, y)

rand_pos(env::NatureEnv) = rand(2) .* size(env) .|> x->ceil(Int,x)
outofbounds(env::NatureEnv, pos::Tuple{Int, Int}) = !all((1,1) .<= pos .<= size(env))

function onehot(idx::Int, len::Int)
    z = zeros(len)
    z[idx] = 1
    z
end


# Clips an n-dimensional vec/tuple between bounds at each dim
function clip(x::Union{Tuple,Vector},mins::Tuple,maxes::Tuple)
    Tuple(clamp(args...) for args in zip(x, mins, maxes))
end

function remove!(matrix::SparseMatrixCSC, args...)
    SparseArrays.dropstored!(matrix, args...)
end

