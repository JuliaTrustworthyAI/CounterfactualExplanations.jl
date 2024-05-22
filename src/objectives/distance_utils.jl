using Flux: Flux

"""
    distance(ce::AbstractCounterfactualExplanation, p::Real=2)

Computes the distance of the counterfactual to the original factual.
"""
function distance(
    ce::AbstractCounterfactualExplanation;
    from::Union{Nothing,AbstractArray}=nothing,
    agg=mean,
    p::Real=1,
    weights::Union{Nothing,AbstractArray}=nothing,
)
    if isnothing(from)
        from = CounterfactualExplanations.factual(ce)
    end
    x′ = CounterfactualExplanations.decode_state(ce)
    if ce.num_counterfactuals == 1
        return LinearAlgebra.norm(x′ .- from, p)
    else
        xs = eachslice(x′; dims=ndims(x′))                      # slices along the last dimension (i.e. the number of counterfactuals)
        if isnothing(weights)
            Δ = agg(map(x′ -> LinearAlgebra.norm(x′ .- from, p), xs))            # aggregate across counterfactuals
        else
            @assert length(weights) == size(first(xs), ndims(first(xs))) "The length of the weights vector must match the number of features."
            Δ = agg(map(x′ -> (LinearAlgebra.norm.(x′ .- from, p)'weights)[1], xs))   # aggregate across counterfactuals
        end
        return Δ
    end
end
