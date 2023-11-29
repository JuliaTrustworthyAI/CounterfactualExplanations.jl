
function generate_counterfactual(
    x::AbstractArray,
    target::RawTargetType,
    data::DataPreprocessing.CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::Generators.GrowingSpheresGenerator;
    num_counterfactuals::Int=1,
    max_iter::Int=1000,
    kwrgs...,
)
    ce = CounterfactualExplanation(
        x, target, data, M, generator; num_counterfactuals, max_iter
    )

    Generators.growing_spheres_generation!(ce)
    Generators.feature_selection!(ce)

    # growing spheres does not support encodings, thus x′ is just s′
    ce.x′ = ce.s′

    return ce
end

"""
    converged(ce::AbstractCounterfactualExplanation)
# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object.
# Returns
- `converged::Bool`:
Finds if we have converged.
"""
function converged(ce::AbstractCounterfactualExplanation)
    model = ce.M
    counterfactual_data = ce.data
    factual = ce.x
    counterfactual = ce.s′

    factual_class = CounterfactualExplanations.Models.predict_label(
        model, counterfactual_data, factual
    )[1]
    counterfactual_class = CounterfactualExplanations.Models.predict_label(
        model, counterfactual_data, counterfactual
    )[1]

    if factual_class == counterfactual_class
        ce.search[:terminated] = true
        ce.search[:converged] = true
    end

    return ce.search[:converged]
end