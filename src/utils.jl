vint = Vector{Int}()
make_frame(x::Int, y::Int) = sparse(vint, vint, vint, x, y)

rand_pos(env::NatureEnv) = rand(2) .* size(env) .|> x->ceil(Int,x)
outofbounds(env::NatureEnv, pos::Tuple{Int, Int}) = !all((1,1) .<= pos .<= size(env))
indices(sp::SparseMatrixCSC) = findall(!iszero, sp)

($)(a::Real, b::Real) = a:sign(b):a+b-sign(b)


# function agent_log(name::String, strings::Vector{String})
#     tb_log = reduce(*, strings)
#     # ex = Meta.parse("@info \"$name\" $tb_log log_step_increment=0")
#     tmp = LineNumberNode(1, Symbol("none"))
#     ex = Expr(:macrocall, Symbol("@info"), tmp, name, [Expr(:(=),Symbol("a",p), p) for p in 1:3]...)
#     eval(ex)
# end


macro agent_log(name, actions)


    quote
    println(actions)
    end
    # println(esc(actions))
    # tb_log = reduce(*, actions)
    # tmp = LineNumberNode(1, Symbol("none"))
    # ex = Expr(:macrocall, Symbol("@info"), tmp, name, [Expr(:(=),Symbol("a",i), actions[i]) for i in length(actions)]...)
    quote
    end
    # return ex

    # println(Meta.@dump(actions))
    # strs = reduce(*, ["a$i=$a " for (i,a) in enumerate(actions)])
    # println("@info $name")
    # reduce(*, ["a$i=$a " for (i,a) in enumerate($actions)])
    # println(strs)
    # Meta.parse("@info \"$name\"")
    # quote
    # eval(Meta.parse("@info \"$name\" $(reduce(*, $strings)) log_step_increment=0"))
    # println("@info \"$name\" $(reduce(*, $(esc(strings)))) log_step_increment=0")
    # end
end

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

