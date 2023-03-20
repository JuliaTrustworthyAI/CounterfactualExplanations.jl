using LinearAlgebra
using SliceMap
using Statistics: mean

"""
    validity(counterfactual_explanation::CounterfactualExplanation; γ=0.5)

Checks of the counterfactual search has been successful with respect to the probability threshold `γ`. In case multiple counterfactuals were generated, the function returns the proportion of successful counterfactuals.
"""
function validity(counterfactual_explanation::CounterfactualExplanation; agg=mean, γ=0.5)
    return agg(CounterfactualExplanations.target_probs(counterfactual_explanation) .>= γ)
end

"""
    validity_strict(counterfactual_explanation::CounterfactualExplanation)

Checks if the counterfactual search has been strictly valid in the sense that it has converged with respect to the pre-specified target probability `γ`.
"""
function validity_strict(counterfactual_explanation::CounterfactualExplanation)
    return validity(
        counterfactual_explanation;
        γ=counterfactual_explanation.convergence[:decision_threshold],
    )
end

"""
    redundancy(counterfactual_explanation::CounterfactualExplanation)

Computes the feature redundancy: that is, the number of features that remain unchanged from their original, factual values.
"""
function redundancy(counterfactual_explanation::CounterfactualExplanation; agg=mean)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    redundant_x = agg(mapslices(x -> sum(x .== 0) / size(x, 1), x′; dims=[1, 2]))
    return redundant_x
end
