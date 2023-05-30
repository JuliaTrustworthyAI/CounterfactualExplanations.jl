using DecisionTree
using LinearAlgebra

"""
    search_path(tree::Union{Leaf, Node}, target::RawTargetType, classes::AbstractArray, path::AbstractArray)

Return a path index list with the inequality symbols, thresholds and feature indices.
"""
function search_path(tree::Union{Leaf, Node}, target::RawTargetType, classes::AbstractArray, path::AbstractArray=[])
    # Check if the current tree is a leaf
    if DecisionTree.is_leaf(tree)
        # Check if the leaf's majority value matches the target
        if tree.majority == tree.majority == classes[target]
            return [path]
        else
            return []
        end
    else
        # Search the left and right subtrees
        paths = []
        append!(paths, search_path(tree.left, target, classes, vcat(path, Dict("inequality_symbol" => 0, 
                                                                      "threshold" => tree.featval,
                                                                      "feature" => tree.featid))))
        append!(paths, search_path(tree.right, target, classes, vcat(path, Dict("inequality_symbol" => 1, 
                                                                       "threshold" => tree.featval,
                                                                       "feature" => tree.featid))))
        return paths
    end
end

"""
    search_path(model::DecisionTreeClassifier, target::RawTargetType, classes::AbstractArray)

Calls `search_path` on the root node of a decision tree.
"""
function search_path(model::DecisionTreeClassifier, target::RawTargetType, classes::AbstractArray)
    return search_path(model.root.node, target, classes)
end

"""
    search_path(model::RandomForestClassifier, target::RawTargetType, classes::AbstractArray)

Calls `search_path` on the root node of a random forest.
"""
function search_path(model::RandomForestClassifier, target::RawTargetType, classes::AbstractArray)
    paths = []
    for tree in model.trees
        append!(paths, search_path(tree, target, classes))
    end
    return paths
end


"""
    feature_tweaking(generator::FeatureTweakGenerator, ensemble::FluxEnsemble, x::AbstractArray, target::RawTargetType)

Returns a counterfactual instance of `x` based on the ensemble of classifiers provided.
"""
function feature_tweaking(generator::HeuristicBasedGenerator, ensemble::Models.TreeModel, x::AbstractArray, target::RawTargetType)
    x_out = deepcopy(x)
    classes = DecisionTree.get_classes(ensemble.model)
    delta = 10^3
    ensemble_prediction = predict_label(ensemble, x)
    for tree in Models.get_individual_classifiers(ensemble)
        classifier = Models.TreeModel(tree, :classification_binary)
        if ensemble_prediction == predict_label(classifier, x) &&
            predict_label(classifier, x) != classes[target]
            
            paths = search_path(classifier.model, target, classes)
            for key in keys(paths)
                path = paths[key]
                es_instance = esatisfactory_instance(generator, x, path)
                if predict_label(classifier, es_instance) == classes[target]
                    if LinearAlgebra.norm(x - es_instance) < delta
                        x_out = es_instance
                        delta = LinearAlgebra.norm(x - es_instance)
                    end
                end
            end
        end
    end
    println("x_out: ", x_out)
    return x_out
end


"""
    esatisfactory_instance(generator::FeatureTweakGenerator, x::AbstractArray, paths::Dict{String, Dict{String, Any}})

Returns an epsilon-satisfactory instance of `x` based on the paths provided.
"""
function esatisfactory_instance(generator::HeuristicBasedGenerator, x::AbstractArray, paths::AbstractArray)
    esatisfactory = deepcopy(x)
    for path in paths
        feature_idx = path["feature"]
        threshold_value = path["threshold"]
        inequality_symbol = path["inequality_symbol"]
        if inequality_symbol == 0
            esatisfactory[feature_idx] = threshold_value - generator.ϵ
        elseif inequality_symbol == 1
            esatisfactory[feature_idx] = threshold_value + generator.ϵ
        else
            println("something wrong")
        end
    end
    println("Esatisfactory: ", esatisfactory)
    return esatisfactory
end
