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

Computes the plausibility of a counterfactual explanation based on the distance from the target. Specifically, the function computes the plausibility as the exponential decay of the distance from the target with rate parameter `λ`. Larger values of `λ` result in a faster decay of the plausibility. If you input data is not normalized, you may want to adjust the rate parameter `λ` accordingly, e.g. higher values for larger feature scales.
"""
function plausibility(
    ce::CounterfactualExplanation,
    fun::typeof(Objectives.distance_from_target);
    K=nothing,
    λ::AbstractFloat=0.5,
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
        K = maximum([1000,size(ce.search[:potential_neighbours], 2)])
    end
    println(K)
    Δ = fun(ce; K=K, kwrgs...)
    return exp_decay(Δ, λ)
end
