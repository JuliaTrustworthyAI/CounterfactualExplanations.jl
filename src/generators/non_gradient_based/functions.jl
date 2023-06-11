using DecisionTree
using DataFrames
using MLJBase
using MLJDecisionTreeInterface

"""
    search_path(tree::Union{Leaf, Node}, target::RawTargetType, path::AbstractArray)

Return a path index list with the inequality symbols, thresholds and feature indices.
"""
function search_path(
    tree::Union{Leaf,DecisionTree.Node}, target::RawTargetType, path::AbstractArray=[]
)
    # Check if the current tree is a leaf
    if DecisionTree.is_leaf(tree)
        # Check if the leaf's majority value matches the target
        if tree.majority == target
            return [path]
        else
            return []
        end
    else
        # Search the left and right subtrees
        paths = []
        append!(
            paths,
            search_path(
                tree.left,
                target,
                vcat(
                    path,
                    Dict(
                        "inequality_symbol" => 0,
                        "threshold" => tree.featval,
                        "feature" => tree.featid,
                    ),
                ),
            ),
        )
        append!(
            paths,
            search_path(
                tree.right,
                target,
                vcat(
                    path,
                    Dict(
                        "inequality_symbol" => 1,
                        "threshold" => tree.featval,
                        "feature" => tree.featid,
                    ),
                ),
            ),
        )
        return paths
    end
end

"""
    search_path(model::DecisionTreeClassifier, target::RawTargetType)

Calls `search_path` on the root node of a decision tree.
"""
function search_path(
    model::MLJDecisionTreeInterface.DecisionTreeClassifier, target::RawTargetType
)
    return search_path(model.root.node, target)
end

"""
    search_path(model::RandomForestClassifier, target::RawTargetType)

Calls `search_path` on the root node of a random forest.
"""
function search_path(
    model::MLJDecisionTreeInterface.RandomForestClassifier, target::RawTargetType
)
    paths = []
    for tree in model.trees
        append!(paths, search_path(tree, target))
    end
    return paths
end

"""
    feature_tweaking(generator::FeatureTweakGenerator, ensemble::FluxEnsemble, x::AbstractArray, target::RawTargetType)

Returns a counterfactual instance of `x` based on the ensemble of classifiers provided.
"""
function feature_tweaking(
    generator::HeuristicBasedGenerator,
    M::Models.TreeModel,
    x::AbstractArray,
    target::RawTargetType,
)
    if predict_label(M, x)[1] == target
        return x
    end

    x_out = deepcopy(x)
    machine = M.model
    delta = 10^3
    # ensemble_prediction = predict_label(M, x)
    fp = MLJBase.fitted_params(machine)
    model = fp.tree.node

    # for tree in Models.get_individual_classifiers(M)
    #     classifier = Models.TreeModel(tree, :classification_binary)
    #     if ensemble_prediction == predict_label(classifier, x) &&
    #         predict_label(classifier, x) != classes[target]

    paths = search_path(model, target)
    for key in keys(paths)
        path = paths[key]
        es_instance = esatisfactory_instance(generator, x, path)
        if target .== predict_label(M, es_instance)[1]
            if LinearAlgebra.norm(x - es_instance) < delta
                x_out = es_instance
                delta = LinearAlgebra.norm(x - es_instance)
            end
        end
    end
    #     end
    # end
    return x_out
end

"""
    esatisfactory_instance(generator::FeatureTweakGenerator, x::AbstractArray, paths::Dict{String, Dict{String, Any}})

Returns an epsilon-satisfactory instance of `x` based on the paths provided.
"""
function esatisfactory_instance(
    generator::HeuristicBasedGenerator, x::AbstractArray, paths::AbstractArray
)
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
    return esatisfactory
end
