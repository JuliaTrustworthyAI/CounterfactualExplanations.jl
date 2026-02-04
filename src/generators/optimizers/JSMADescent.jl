using Optimisers: Optimisers

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
    mutability::Union{Nothing,AbstractArray}=nothing,
)
    if all(isnothing.([η, n]))
        η = 0.1
        n = 10
    elseif isnothing(η) && !isnothing(n)
        η = 1 / n
    elseif !isnothing(η) && isnothing(n)
        n = Int(maximum([1, 1 / η]))
    end
    return JSMADescent(η, n, mutability)
end

# Initialize state - just return zeros array
function Optimisers.init(o::JSMADescent, x::AbstractArray)
    return zeros(size(x))
end

# Apply the optimization rule
function Optimisers.apply!(o::JSMADescent, times_changed, x, Δ)

    # Mutability:
    if !isnothing(o.mutability)
        Δ[o.mutability .== :none] .= 0
    end

    # Times changed:
    Δ[times_changed .== o.n] .= 0

    # Helper function to choose most salient:
    function choose_most_salient(x)
        s = (abs.(x) .== maximum(abs.(x); dims=1)) .* sign.(x)
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
    Δ = mapslices(x -> choose_most_salient(x), Δ; dims=(1, 2)) # choose most salient feature

    # Update times_changed
    new_times_changed = times_changed .+ (Δ .!= 0.0)
    new_times_changed = mapslices(
        x -> all(x .== o.n) ? zeros(size(x)) : x, new_times_changed; dims=(1, 2)
    )

    Δ = o.eta / size(x, 3) .* Δ

    return new_times_changed, Δ
end
