using CounterfactualExplanations.Convergence: threshold_reached
using Statistics: Statistics

include("faithfulness/faithfulness.jl")
include("plausibility/plausibility.jl")

"""
    validity(ce::CounterfactualExplanation; γ=0.5)

Checks of the counterfactual search has been successful in that the predicted label corresponds to the specified target. In case multiple counterfactuals were generated, the function returns the proportion of successful counterfactuals.
"""
function validity(
    ce::CounterfactualExplanation;
    agg=Statistics.mean,
    threshold::Union{Nothing,AbstractFloat}=nothing,
)
    val = if isnothing(threshold)
        agg(CounterfactualExplanations.counterfactual_label(ce) .== ce.target)
    else
        threshold_reached(ce) |> x -> float.(x)
    end
    return val
end

"""
    validity_strict(ce::CounterfactualExplanation; kwrgs...)

Checks if the counterfactual search has been strictly valid in the sense that it has converged with respect to the pre-specified target probability `γ`.
"""
function validity_strict(ce::CounterfactualExplanation; kwrgs...)
    return validity(ce; threshold=ce.convergence.decision_threshold, kwrgs...)
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

"""
    feature_sensitivity(
        ce::AbstractCounterfactualExplanation, d::Union{Int,Vector{Int}}=[1]; kwrgs...
    )

Return the sensitivity to feature(s) `d` in terms of absolute changes associated with the counterfactual. Any keyword arguments accepted by [`distance`](@ref) can be passed to `kwrgs...`.
"""
function feature_sensitivity(
    ce::AbstractCounterfactualExplanation, d::Union{Int,Vector{Int}}=[1]; kwrgs...
)
    if isa(d, Int)
        d = [d]
    end
    return [distance(ce; d=[d], kwrgs...) for d in d]
end
