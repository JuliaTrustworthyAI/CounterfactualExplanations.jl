using CounterfactualExplanations.Objectives

function plausibility(ce::AbstractCounterfactualExplanation; kwrgs...)
    return plausibility(ce, Objectives.EnergyDifferential(); kwrgs...)
end

"""
    plausibility(
        ce::CounterfactualExplanation,
        fun::typeof(Objectives.distance_from_target);
        K=nothing,
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

    if isnothing(K)
        K = minimum([1000, size(ce.search[:potential_neighbours], 2)])
    end

    # Compute the distance from the target:
    Δ = fun(ce; K=K, kwrgs...)
    return -Δ
end

"""
    plausibility(
        ce::CounterfactualExplanation,
        fun::typeof(Objectives.distance_from_target);
        K=nothing,
        kwrgs...,
    )

Computes the plausibility of a counterfactual explanation based on the cosine similarity between the counterfactual and samples drawn from the target distribution.
"""
function plausibility(
    ce::CounterfactualExplanation, fun::Objectives.EnergyDifferential; kwrgs...
)
    Δ = fun(ce)
    return -Δ
end
