using CounterfactualExplanations.Objectives

function plausibility(ce::AbstractCounterfactualExplanation; kwrgs...)
    return plausibility(ce, Objectives.distance_from_target; kwrgs...)
end

"""
    plausibility(
        ce::CounterfactualExplanation,
        fun::typeof(Objectives.distance_from_target);
        λ::AbstractFloat=1.0,
        kwrgs...,
    )

Computes the plausibility of a counterfactual explanation based on the distance from the target. Specifically, the function computes the plausibility as the exponential of the CDF of the distance from the target.
"""
function plausibility(
    ce::CounterfactualExplanation,
    fun::typeof(Objectives.distance_from_target);
    λ::AbstractFloat=1.0,
    kwrgs...,
)
    # If the potential neighbours have not been computed, do so:
    get!(
        ce.search,
        :potential_neighbours,
        CounterfactualExplanations.find_potential_neighbours(ce),
    )
    # Compute the distance from the target:
    Δ = fun(ce; kwrgs...)
    return exp_decay(Δ, λ)
end
