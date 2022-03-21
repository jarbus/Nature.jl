vint = Vector{Int}()
make_frame(x::Int, y::Int) = sparse(vint, vint, vint, x, y)

rand_pos(env::NatureEnv) = rand(2) .* size(env) .|> x->ceil(Int,x)
outofbounds(env::NatureEnv, pos::Tuple{Int, Int}) = !all((1,1) .<= pos .<= size(env))
indices(sp::SparseMatrixCSC) = findall(!iszero, sp)

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

function remove!(matrix::SparseMatrixCSC, args...)
    SparseArrays.dropstored!(matrix, args...)
end

