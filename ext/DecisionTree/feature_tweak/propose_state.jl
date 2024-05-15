"""
     propose_state(
        generator::Generators.FeatureTweakGenerator, ce::AbstractCounterfactualExplanation
    )

Overloads the [`propose_state`](@ref) method for the `FeatureTweakGenerator`.
"""
function propose_state(
    generator::Generators.FeatureTweakGenerator, ce::AbstractCounterfactualExplanation
)
    delta = 10^3
    ensemble_prediction = Models.predict_label(ce.M, ce.x)[1]

    for classifier in Models.get_individual_classifiers(ce.M)
        if ensemble_prediction != ce.target
            y_levels = MLJBase.classes(
                MLJBase.predict(ce.M.model, DataFrames.DataFrame(ce.x', :auto))
            )
            paths = search_path(classifier, y_levels, ce.target)

            for key in keys(paths)
                path = paths[key]
                es_instance = esatisfactory_instance(ce.generator, ce.x, path)
                if ce.target .== Models.predict_label(ce.M, es_instance)[1]
                    s′_old = ce.s′
                    ce.s′ = reshape(es_instance, :, 1)
                    new_delta = calculate_delta(ce)
                    if new_delta < delta
                        delta = new_delta
                    else
                        ce.s′ = s′_old
                    end
                end
            end
        end
    end

    return ce.s′
end