using ..CounterfactualExplanations
using LinearAlgebra
using SliceMap

"""
    distance(counterfactual_explanation::AbstractCounterfactualExplanation, p::Real=2)

Computes the distance of the counterfactual to the original factual.
"""
function distance(counterfactual_explanation::AbstractCounterfactualExplanation, p::Real=2)
    x = CounterfactualExplanations.factual(counterfactual_explanation)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    Δ = SliceMap.slicemap(_x -> permutedims([LinearAlgebra.norm(_x .- x, p)]), x′, dims=(1, 2))
    return Δ
end

"""
    distance_l0(counterfactual_explanaation::AbstractCounterfactualExplanation)

Computes the L0 distance of the counterfactual to the original factual.
"""
distance_l0(counterfactual_explanation::AbstractCounterfactualExplanation) = distance(counterfactual_explanation, 0)

"""
    distance_l1(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L1 distance of the counterfactual to the original factual.
"""
distance_l1(counterfactual_explanation::AbstractCounterfactualExplanation) = distance(counterfactual_explanation, 1)

"""
    distance_l2(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L2 (Euclidean) distance of the counterfactual to the original factual.
"""
distance_l2(counterfactual_explanation::AbstractCounterfactualExplanation) = distance(counterfactual_explanation, 2)

"""
    distance_linf(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L-inf distance of the counterfactual to the original factual.
"""
distance_linf(counterfactual_explanation::AbstractCounterfactualExplanation) = distance(counterfactual_explanation, Inf)