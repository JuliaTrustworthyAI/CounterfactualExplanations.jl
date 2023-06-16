"""
    FeatureTweakGenerator(; ϵ::AbstractFloat, kwargs...)

Constructs a new Feature Tweak Generator object.

Uses the L2-norm as the penalty to measure the distance between the counterfactual and the factual.
According to the paper by Tolomei er al., an alternative choice here would be using the L0-norm to simply minimize the number of features that are changed through the tweak.

# Arguments
- `ϵ::AbstractFloat`: The tolerance value for the feature tweaks. Described at length in Tolomei et al. (https://arxiv.org/pdf/1706.06691.pdf).

# Returns
- `generator::HeuristicBasedGenerator`: A non-gradient-based generator that can be used to generate counterfactuals using the feature tweak method.
"""
function FeatureTweakGenerator(; ϵ::AbstractFloat=0.1, kwargs...)
    return HeuristicBasedGenerator(; penalty=Objectives.distance_l2, ϵ=ϵ, kwargs...)
end

"""
    feature_tweaking(generator::FeatureTweakGenerator, ensemble::FluxEnsemble, x::AbstractArray, target::RawTargetType)

Returns a counterfactual instance of `x` based on the ensemble of classifiers provided.

# Arguments
- `generator::FeatureTweakGenerator`: The feature tweak generator.
- `M::Models.TreeModel`: The model for which the counterfactual is generated. Must be a tree-based model.
- `x::AbstractArray`: The factual instance.
- `target::RawTargetType`: The target class.

# Returns
- `x_out::AbstractArray`: The counterfactual instance.

# Example
x = feature_tweaking(generator, M, x, target) # returns a counterfactual instance of `x` based on the ensemble of classifiers provided
"""
function feature_tweaking(
    generator::HeuristicBasedGenerator,
    M::Models.TreeModel,
    x::AbstractArray,
    target::RawTargetType,
)
    if !(M isa Models.TreeModel)
        throw(
            ArgumentError(
                "The feature tweak generator is only available for tree-based models."
            ),
        )
    end
    if M.likelihood == :classification_multi
        throw(
            ArgumentError(
                "The feature tweak generator is only available for binary classification problems.",
            ),
        )
    end

    if Models.predict_label(M, x)[1] == target
        return x
    end

    x_out = deepcopy(x)
    machine = M.model
    delta = 10^3
    ensemble_prediction = Models.predict_label(M, x)[1]
    fp = MLJBase.fitted_params(machine)
    model = fp.tree.node

    for classifier in Models.get_individual_classifiers(M)
        if ensemble_prediction == Models.predict_label(classifier, x)[1] &&
            Models.predict_label(classifier, x)[1] != target
            paths = search_path(model, target)
            for key in keys(paths)
                path = paths[key]
                es_instance = esatisfactory_instance(generator, x, path)
                if target .== Models.predict_label(M, es_instance)[1]
                    if LinearAlgebra.norm(x - es_instance) < delta
                        x_out = es_instance
                        delta = LinearAlgebra.norm(x - es_instance)
                    end
                end
            end
        end
    end
    return x_out
end

"""
    esatisfactory_instance(generator::FeatureTweakGenerator, x::AbstractArray, paths::Dict{String, Dict{String, Any}})

Returns an epsilon-satisfactory counterfactual for `x` based on the paths provided.

# Arguments
- `generator::FeatureTweakGenerator`: The feature tweak generator.
- `x::AbstractArray`: The factual instance.
- `paths::Dict{String, Dict{String, Any}}`: A list of paths to the leaves of the tree to be used for tweaking the feature.

# Returns
- `esatisfactory::AbstractArray`: The epsilon-satisfactory instance.

# Example
esatisfactory = esatisfactory_instance(generator, x, paths) # returns an epsilon-satisfactory counterfactual for `x` based on the paths provided
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
            error("Unable to find a valid e-satisfactory instance.")
        end
    end
    return esatisfactory
end

"""
    search_path(tree::Union{Leaf, Node}, target::RawTargetType, path::AbstractArray)

Return a path index list with the inequality symbols, thresholds and feature indices.

# Arguments
- `tree::Union{Leaf, Node}`: The root node of a decision tree.
- `target::RawTargetType`: The target class.
- `path::AbstractArray`: A list containing the paths found thus far.

# Returns
- `paths::AbstractArray`: A list of paths to the leaves of the tree to be used for tweaking the feature.

# Example
paths = search_path(tree, target) # returns a list of paths to the leaves of the tree to be used for tweaking the feature
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

# Arguments
- `model::DecisionTreeClassifier`: The decision tree model.
- `target::RawTargetType`: The target class.

# Returns
- `paths::AbstractArray`: A list of paths to the leaves of the tree to be used for tweaking the feature.

# Example
paths = search_path(model, target) # returns a list of paths to the leaves of the tree to be used for tweaking the feature
"""
function search_path(
    model::MLJDecisionTreeInterface.DecisionTreeClassifier, target::RawTargetType
)
    return search_path(model.root.node, target)
end

"""
    search_path(model::RandomForestClassifier, target::RawTargetType)

Calls `search_path` on the root node of a random forest.

# Arguments
- `model::RandomForestClassifier`: The random forest model.
- `target::RawTargetType`: The target class.

# Returns
- `paths::AbstractArray`: A list of paths to the leaves of the tree to be used for tweaking the feature.

# Example
paths = search_path(model, target) # returns a list of paths to the leaves of the tree to be used for tweaking the feature
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
