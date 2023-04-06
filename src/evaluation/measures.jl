using LinearAlgebra
using SliceMap
using Statistics: mean

"""
    validity(ce::CounterfactualExplanation; γ=0.5)

Checks of the counterfactual search has been successful with respect to the probability threshold `γ`. In case multiple counterfactuals were generated, the function returns the proportion of successful counterfactuals.
"""
function validity(ce::CounterfactualExplanation; agg=mean, γ=0.5)
    return agg(CounterfactualExplanations.target_probs(ce) .>= γ)
end

"""
    validity_strict(ce::CounterfactualExplanation)

Checks if the counterfactual search has been strictly valid in the sense that it has converged with respect to the pre-specified target probability `γ`.
"""
function validity_strict(ce::CounterfactualExplanation)
    return validity(
        ce;
        γ=ce.convergence[:decision_threshold],
    )
end

"""
    redundancy(ce::CounterfactualExplanation)

Computes the feature redundancy: that is, the number of features that remain unchanged from their original, factual values.
"""
function redundancy(ce::CounterfactualExplanation; agg=mean)
    x′ = CounterfactualExplanations.counterfactual(ce)
    redundant_x = agg(mapslices(x -> sum(x .== 0) / size(x, 1), x′; dims=[1, 2]))
    return redundant_x
end
