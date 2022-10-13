using LinearAlgebra
using Statistics

"""
    success_rate(counterfactual_explanation::CounterfactualExplanation; agg=mean)

Computes the success rate.
"""
function success_rate(counterfactual_explanation::CounterfactualExplanation; agg=mean)
    agg(Counterfactuals.target_probs(counterfactual_explanation) .>= counterfactual_explanation.params[:γ])
end   

function success_rate(counterfactual_explanations::Vector{CounterfactualExplanation}; agg=mean)
    agg(success_rate.(counterfactual_explanations))
end    

"""
    distance(counterfactual_explanation::CounterfactualExplanation; agg=mean)

Computes the distance.
"""
function distance(counterfactual_explanation::CounterfactualExplanation; agg=mean)
    x = Counterfactuals.factual(counterfactual_explanation)
    x′ = Counterfactuals.counterfactual(counterfactual_explanation)
    return agg(LinearAlgebra.norm.(x.-x′))
end

function distance(counterfactual_explanations::Vector{CounterfactualExplanation}; agg=mean)
    agg(distance.(counterfactual_explanations))
end

"""
    redundancy(counterfactual_explanation::CounterfactualExplanation; agg=mean)

Computes the feature redundancy.
"""
function redundancy(counterfactual_explanation::CounterfactualExplanation; agg=mean)
    x′ = Counterfactuals.counterfactual(counterfactual_explanation)
    redundant_x = mapslices(x -> sum(x .== 0)/size(x,1), x′, dims=[1,2])
    return agg(redundant_x)
end

function redundancy(counterfactual_explanations::Vector{CounterfactualExplanation}; agg=mean)
    agg(redundancy.(counterfactual_explanations))
end

