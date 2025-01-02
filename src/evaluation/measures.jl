using Statistics: Statistics

include("faithfulness/faithfulness.jl")
include("plausibility/plausibility.jl")

"""
    validity(ce::CounterfactualExplanation; γ=0.5)

Checks of the counterfactual search has been successful in that the predicted label corresponds to the specified target. In case multiple counterfactuals were generated, the function returns the proportion of successful counterfactuals.
"""
function validity(ce::CounterfactualExplanation; agg=Statistics.mean, γ=0.5)
    val = agg(CounterfactualExplanations.counterfactual_label(ce) .== ce.target)
    val = val isa LinearAlgebra.AbstractMatrix ? vec(val) : val
    return val
end

"""
    validity_strict(ce::CounterfactualExplanation)

Checks if the counterfactual search has been strictly valid in the sense that it has converged with respect to the pre-specified target probability `γ`.
"""
function validity_strict(ce::CounterfactualExplanation)
    return validity(ce; γ=ce.convergence.decision_threshold)
end

"""
    redundancy(ce::CounterfactualExplanation)

Computes the feature redundancy: that is, the number of features that remain unchanged from their original, factual values.
"""
function redundancy(ce::CounterfactualExplanation; agg=Statistics.mean, tol=1e-5)
    cf = CounterfactualExplanations.counterfactual(ce)
    redundant_x = [
        agg(sum(abs.(x .- ce.factual) .< tol) / size(x, 1)) for
        x in eachslice(cf; dims=ndims(cf))
    ]
    redundant_x = length(redundant_x) == 1 ? redundant_x[1] : redundant_x
    return redundant_x
end
