
Returns an epsilon-satisfactory instance of `x` based on the paths provided.
"""
function esatisfactory_instance(generator::FeatureTweakGenerator, x, epsilon, paths)
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