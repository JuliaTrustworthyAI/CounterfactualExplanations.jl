using LinearAlgebra
using Statistics

"""
    success_rate(counterfactual_explanation::CounterfactualExplanation; agg=mean)

Checks of the counterfactual search has been successful. In case multiple counterfactuals were generated, the function returns the proportion of successful counterfactuals.
"""
function success_rate(counterfactual_explanation::CounterfactualExplanation; agg = mean)
    agg(
        CounterfactualExplanations.target_probs(counterfactual_explanation) .>=
        counterfactual_explanation.params[:γ],
    )
end

"""
    success_rate(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        agg = mean,
    )

Computes the proportion of successful counterfactuals acress a vector of counterfactual explanations.
"""
function success_rate(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    agg = mean,
)
    agg(success_rate.(counterfactual_explanations))
end

"""
    distance(counterfactual_explanation::CounterfactualExplanation; agg=mean)

Computes the Euclidean distance of the counterfactual to the original factual.
"""
function distance(counterfactual_explanation::CounterfactualExplanation; agg = mean)
    x = CounterfactualExplanations.factual(counterfactual_explanation)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    return agg(LinearAlgebra.norm.(x .- x′))
end

"""
    distance(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        agg=mean
    )

Computes the average Euclidean distance of multiple counterfactuals from their corresponding factuals.
"""
function distance(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    agg = mean,
)
    agg(distance.(counterfactual_explanations))
end

"""
    redundancy(counterfactual_explanation::CounterfactualExplanation; agg=mean)

Computes the feature redundancy: that is, the number of features that remain unchanged from their original, factual values.
"""
function redundancy(counterfactual_explanation::CounterfactualExplanation; agg = mean)
    x′ = CounterfactualExplanations.counterfactual(counterfactual_explanation)
    redundant_x = mapslices(x -> sum(x .== 0) / size(x, 1), x′, dims = [1, 2])
    return agg(redundant_x)
end

"""
    redundancy(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        agg=mean
    )

Computes the average redundancy across multiple counterfactuals.
"""
function redundancy(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    agg = mean,
)
    agg(redundancy.(counterfactual_explanations))
end
