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
    return zeros(eltype(x), size(x))
end

# Apply the optimization rule
function Optimisers.apply!(o::JSMADescent, times_changed, x, Δ)

    # Mutability:
    if !isnothing(o.mutability)
        Δ[o.mutability .== :none] .= 0.0
    end

    # Times changed:
    Δ[times_changed .== o.n] .= 0

    # Choose most salient - use similar(Δ) to preserve exact type
    n_samples = size(Δ, 3)
    Δ_out = similar(Δ)  # This preserves the exact type of Δ
    fill!(Δ_out, 0)

    for sample_idx in 1:n_samples
        sample_slice = Δ[:, :, sample_idx]

        # Find maximum absolute value
        max_val = maximum(abs.(sample_slice))

        # Find all positions with maximum value
        s = (abs.(sample_slice) .== max_val) .* sign.(sample_slice)
        non_zero_elements = findall(s .!= 0)

        # If more than one equal, choose first (deterministic)
        if length(non_zero_elements) > 0
            keep_idx = first(non_zero_elements)
            Δ_out[keep_idx, sample_idx] = s[keep_idx]
        end
    end

    Δ = Δ_out

    # Update times_changed
    new_times_changed = times_changed .+ (Δ .!= 0.0)

    for sample_idx in 1:n_samples
        sample_slice = @view new_times_changed[:, :, sample_idx]
        if all(sample_slice .== o.n)
            sample_slice .= 0
        end
    end

    # Scale - convert to match Δ's type
    scale_factor = convert(eltype(Δ), o.eta / size(Δ, 3))
    Δ = scale_factor .* Δ

    return new_times_changed, Δ
end
