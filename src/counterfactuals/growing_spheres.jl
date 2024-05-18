
"""
    generate_counterfactual(
        x::Matrix,
        target::RawTargetType,
        data::DataPreprocessing.CounterfactualData,
        M::Models.AbstractModel,
        generator::Generators.GrowingSpheresGenerator;
        num_counterfactuals::Int=1,
        convergence::Union{AbstractConvergence,Symbol}=Convergence.DecisionThresholdConvergence(;
            decision_threshold=(1 / length(data.y_levels)), max_iter=1000
        ),
        kwrgs...,
    )

Overloads the `generate_counterfactual` for the `GrowingSpheresGenerator` generator.
"""
function generate_counterfactual(
    x::Matrix,
    target::RawTargetType,
    data::DataPreprocessing.CounterfactualData,
    M::Models.AbstractModel,
    generator::Generators.GrowingSpheresGenerator;
    num_counterfactuals::Int=1,
    convergence::Union{AbstractConvergence,Symbol}=Convergence.DecisionThresholdConvergence(;
        decision_threshold=(1 / length(data.y_levels)), max_iter=1000
    ),
    kwrgs...,
)
    ce = CounterfactualExplanation(
        x, target, data, M, generator; num_counterfactuals, convergence
    )

    Generators.growing_spheres_generation!(ce)
    Generators.feature_selection!(ce)

    # growing spheres does not support encodings, thus x′ is just s′
    ce.x′ = ce.s′

    return ce
end
