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

Computes the plausibility of a counterfactual explanation based on the cosine similarity between the counterfactual and samples drawn from the target distribution.
"""
function plausibility(
    ce::CounterfactualExplanation,
    fun::typeof(Objectives.distance_from_target);
    K=nothing,
    kwrgs...,
)
    # If the potential neighbours have not been computed, do so:
    get!(
        ce.search,
        :potential_neighbours,
        CounterfactualExplanations.find_potential_neighbours(ce),
    )
    # Compute the distance from the target:
    if isnothing(K)
        K = maximum([1000, size(ce.search[:potential_neighbours], 2)])
    end
    Δ = fun(ce; K=K, kwrgs...)
    return -Δ
end
