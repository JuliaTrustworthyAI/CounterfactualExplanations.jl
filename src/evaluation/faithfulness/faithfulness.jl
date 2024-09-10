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

Computes the faithfulness of a counterfactual explanation based on the cosine similarity between the counterfactual and samples drawn from the model posterior through SGLD (see [`distance_from_posterior`](@ref)).
"""
function faithfulness(
    ce::CounterfactualExplanation,
    fun::typeof(distance_from_posterior);
    kwrgs...,
)
    Δ = fun(ce; kwrgs...)
    return -Δ
end
