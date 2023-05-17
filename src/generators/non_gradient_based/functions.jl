using DecisionTree

"""
    search_path(classifier::TreeModel, num_models::Int, target::RawTargetType)

Return a path index list with the ids of the leaf nodes, inequality symbols, thresholds and feature indices
"""
function search_path(classifier::Models.TreeModel, num_models::Int, target::RawTargetType)
    println(DecisionTree.get_classes(classifier.model))
    search_path(classifier.model.root, target, DecisionTree.get_classes(classifier.model))
end


function search_path(tree::Union{DecisionTree.Leaf, DecisionTree.Node}, target, classes::AbstractArray, path=[])
    # Check if the current tree is a leaf
    
    if DecisionTree.is_leaf(tree)
        # Check if the leaf's majority value matches the target
        println(tree.majority == classes[target + 1])
        if tree.majority == classes[target + 1]
            return [path]
        else
            return []
        end
    else
        # Search the left and right subtrees
        paths = []
        append!(paths, search_path(tree.left, target, classes, vcat(path, ("L", tree.featid, tree.featval))))
        append!(paths, search_path(tree.right, target, classes, vcat(path, ("R", tree.featid, tree.featval))))
        return paths
    end
end

function search_path(root::DecisionTree.Root, target, classes::AbstractArray)
    return search_path(root.node, target, classes)
end

function search_path(ensemble::DecisionTree.Ensemble, target)
    paths = []
    for tree in ensemble.trees
        append!(paths, search_path(tree, target))
    end
    return paths
end

"""
    feature_tweaking(generator::FeatureTweakGenerator, ensemble::FluxEnsemble, x::AbstractArray, target::RawTargetType)

Returns a counterfactual instance of `x` based on the ensemble of classifiers provided.
"""
function feature_tweaking(generator::HeuristicBasedGenerator, ensemble::Models.TreeModel, x::AbstractArray, target::RawTargetType)
    x_out = deepcopy(x)
    delta = 10^3
    ensemble_prediction = predict_label(ensemble, x)
    for tree in Models.get_individual_classifiers(ensemble)
        classifier = Models.TreeModel(tree, :classification_binary)
        if ensemble_prediction == predict_label(classifier, x) &&
            predict_label(classifier, x) != target
            
            paths = search_path(classifier, length(Models.get_individual_classifiers(ensemble)), target)
            println(paths)
            for key in keys(paths)
                path = paths[key]
                es_instance = esatisfactory_instance(generator, x, path)
                println(typeof(es_instance))
                if predict_label(classifier, es_instance) == target
                    if generator.loss(x, es_instance) < delta
                        x_out = es_instance
                        delta = generator.penalty(x, es_instance)
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
function esatisfactory_instance(generator::HeuristicBasedGenerator, x::AbstractArray, paths::Dict{String, Dict{String, Any}})
    esatisfactory = deepcopy(x)
    for i in 1:length(paths["feature"])
        feature_idx = paths["feature"][i]
        threshold_value = paths["threshold"][i]
        inequality_symbol = paths["inequality_symbol"][i]
        if inequality_symbol == 0
            esatisfactory[feature_idx] = threshold_value - generator.ϵ
        elseif inequality_symbol == 1
            esatisfactory[feature_idx] = threshold_value + generator.ϵ
        else
            println("something wrong")
        end
    end
    return esatisfactory
end
