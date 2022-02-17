vint = Vector{Int32}()
make_frame(x::Int, y::Int) = sparse(vint, vint, vint, x, y)

# Clips an n-dimensional vec/tuple between bounds at each dim
function clip(x::Union{Tuple,Vector},mins::Tuple,maxes::Tuple)
    Tuple(clamp(args...) for args in zip(x, mins, maxes))
end

