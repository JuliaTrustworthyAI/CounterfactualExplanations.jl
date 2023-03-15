using Optimisers
using SliceMap

"An optimisation rule that can be used to implement a Jacobian-based Saliency Map Attack."
struct JSMADescent <: Optimisers.AbstractRule
    eta::AbstractFloat
    n::Int
    mutability::Union{Nothing,AbstractArray}
end

"Outer constructor for the [`JSMADescent`](@ref) rule."
function JSMADescent(;
    η::Union{Nothing,AbstractFloat}=nothing, 
    n::Union{Nothing,Int}=nothing, 
    mutability::Union{Nothing,AbstractArray}=nothing
)
    if all(isnothing.([η, n]))
        η = 0.1
        n = 10
    elseif isnothing(η) && !isnothing(n)
        η = 1 / n
    elseif !isnothing(η) && isnothing(n)
        n = 1 / eta
    end
    return JSMADescent(η, n, mutability)
end

function Optimisers.apply!(o::JSMADescent, state, params, Δ)

    # Mutability:
    if !isnothing(o.mutability)
        Δ[o.mutability.==:none] .= 0
    end

    # Helper function to choose most salient:
    function choose_most_salient(x)
        s = -((abs.(x) .== maximum(abs.(x), dims=1)) .* sign.(x))
        non_zero_elements = findall(vec(s) .!= 0)
        # If more than one equal, randomise:
        if length(non_zero_elements) > 1
            keep_ = rand(non_zero_elements)
            s_ = zeros(size(s))
            s_[keep_] = s[keep_]
            s = s_
        end
        return s
    end

    # Updating:
    Δ = SliceMap.slicemap(x -> choose_most_salient(x), Δ, dims=(1,2)) # choose most salient feature
    nextstate = state .+ (Δ .!= 0.0)
    nextstate = SliceMap.slicemap(x -> all(x .== o.n) ? zeros(size(x)) : x, nextstate, dims=(1,2))
    newΔ = o.eta .* Δ

    return nextstate, newΔ
end

function Optimisers.init(o::JSMADescent, x::AbstractArray)
    times_changed = zeros(size(x))
    return times_changed
end