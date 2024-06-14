include("utils.jl")

using CounterfactualExplanations.Objectives

function faithfulness(ce::AbstractCounterfactualExplanation; kwrgs...)
    return faithfulness(ce, distance_from_posterior; kwrgs...)
end

"""
    faithfulness(
        ce::CounterfactualExplanation,
        fun::typeof(Objectives.distance_from_target);
        λ::AbstractFloat=1.0,
        kwrgs...,
    )

Computes the faithfulness of a counterfactual explanation based on the distance from the target. Specifically, the function computes the faithfulness as the exponential decay of the distance from the samples drawn from the learned posterior of the model with rate parameter `λ`. Larger values of `λ` result in a faster decay of the faithfulness. If you input data is not normalized, you may want to adjust the rate parameter `λ` accordingly, e.g. higher values for larger feature scales.
"""
function faithfulness(
    ce::CounterfactualExplanation,
    fun::typeof(distance_from_posterior);
    λ::AbstractFloat=0.9,
    normalize::Bool=true,
    kwrgs...,
)
    Δ = fun(ce; kwrgs...)
    if normalize
        Δ = Δ / size(ce.x, 1)
    end
    return exp_decay(Δ, λ)
end
