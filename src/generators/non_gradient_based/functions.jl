"""
    feature_tweaking(ensemble::FluxEnsemble, input_data::CounterfactualData, x, class_labels, target, epsilon, loss)

Returns a counterfactual instance of `x` based on the ensemble of classifiers provided.
"""
function feature_tweaking(ensemble::FluxEnsemble, input_data::CounterfactualData, x, class_labels, target, epsilon, loss)
    x_out = deepcopy(x)
    delta = 10^3
    for classifier in ensemble
        if predict_label(ensemble, input_data, x) == predict_label(classifier, input_data, x) &&
            predict_label(classifier, input_data, x) != target
            paths = search_path(classifier, class_labels, target)
            for key in keys(paths)
                path = paths[key]
                es_instance = esatisfactory_instance(x, epsilon, path)
                if predict_label(classifier, input_data, es_instance) == target
                    if loss(x, es_instance) < delta
                        x_out = es_instance
                        delta = loss(x, es_instance)
                    end
                end
            end
        end
    end
    return x_out
end

"""
    esatisfactory_instance(x, epsilon, paths)

Returns an epsilon-satisfactory instance of `x` based on the paths provided.
"""
function esatisfactory_instance(x, epsilon, paths)
    esatisfactory = deepcopy(x)
    for i in 1:length(paths["feature"])
        feature_idx = paths["feature"][i]
        threshold_value = paths["threshold"][i]
        inequality_symbol = paths["inequality_symbol"][i]
        if inequality_symbol == 0
            esatisfactory[feature_idx] = threshold_value - epsilon
        elseif inequality_symbol == 1
            esatisfactory[feature_idx] = threshold_value + epsilon
        else
            println("something wrong")
        end
    end
    return esatisfactory
end