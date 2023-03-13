using ..CounterfactualExplanations
using LinearAlgebra
using SliceMap
using Statistics: mean

"""
    distance(counterfactual_explanation::AbstractCounterfactualExplanation, p::Real=2)

Computes the distance of the counterfactual to the original factual.
"""
function distance(counterfactual_explanation::AbstractCounterfactualExplanation, p::Real=2; agg=mean)
    x = CounterfactualExplanations.factual(counterfactual_explanation)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    Δ = agg(SliceMap.slicemap(_x -> permutedims([norm(_x .- x, p)]), x′, dims=(1, 2)))
    return Δ
end

"""
    distance_l0(counterfactual_explanaation::AbstractCounterfactualExplanation)

Computes the L0 distance of the counterfactual to the original factual.
"""
distance_l0(counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean) = distance(counterfactual_explanation, 0; agg=agg)

"""
    distance_l1(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L1 distance of the counterfactual to the original factual.
"""
distance_l1(counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean) = distance(counterfactual_explanation, 1, agg=agg)

"""
    distance_l2(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L2 (Euclidean) distance of the counterfactual to the original factual.
"""
distance_l2(counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean) = distance(counterfactual_explanation, 2; agg=agg)

"""
    distance_linf(counterfactual_explanation::AbstractCounterfactualExplanation)

Computes the L-inf distance of the counterfactual to the original factual.
"""
distance_linf(counterfactual_explanation::AbstractCounterfactualExplanation; agg=mean) = distance(counterfactual_explanation, Inf; agg=agg)

"""
    ddp_diversity(
        counterfactual_explanation::AbstractCounterfactualExplanation;
        perturbation_size=1e-5
    )

Evaluates how diverse the counterfactuals are using a Determintal Point Process (DDP).
"""
function ddp_diversity(
    counterfactual_explanation::AbstractCounterfactualExplanation;
    perturbation_size=1e-5,
    agg=det,
)
    X = counterfactual_explanation.s′
    xs = eachslice(X, dims=ndims(X))
    K = [1 / (1 + norm(x .- y)) for x in xs, y in xs]
    K += LinearAlgebra.Diagonal(randn(eltype(X), size(X, 3)) * convert(eltype(X), perturbation_size))
    cost = -agg(K)
    return cost
end

