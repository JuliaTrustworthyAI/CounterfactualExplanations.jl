"""
    calculate_delta(ce::AbstractCounterfactualExplanation, penalty::Vector{Function})

Calculates the penalty for the proposed feature tweak.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object.

# Returns
- `delta::Float64`: The calculated penalty for the proposed feature tweak.
"""
function calculate_delta(ce::AbstractCounterfactualExplanation)
    penalty = ce.generator.penalty
    penalty_functions = penalty isa Function ? [penalty] : penalty
    delta = sum([p(ce) for p in penalty_functions])
    return delta
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
    generator::FeatureTweakGenerator, x::AbstractArray, paths::AbstractArray
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
            error("Unable to find a valid ϵ-satisfactory instance.")
        end
    end

    return esatisfactory
end

"""
    search_path(tree::Union{DecisionTree.Leaf, DecisionTree.Node}, target::RawTargetType, path::AbstractArray)

Return a path index list with the inequality symbols, thresholds and feature indices.

# Arguments
- `tree::Union{DecisionTree.Leaf, DecisionTree.Node}`: The root node of a decision tree.
- `target::RawTargetType`: The target class.
- `path::AbstractArray`: A list containing the paths found thus far.

# Returns
- `paths::AbstractArray`: A list of paths to the leaves of the tree to be used for tweaking the feature.

# Example
paths = search_path(tree, target) # returns a list of paths to the leaves of the tree to be used for tweaking the feature
"""
function search_path(
    tree::Union{DecisionTree.Leaf,DecisionTree.Node},
    y_levels::AbstractArray,
    target::RawTargetType,
    path::AbstractArray=[],
)
    # Check if the current tree is a leaf
    if DecisionTree.is_leaf(tree)
        # Check if the leaf's majority value matches the target
        if y_levels[tree.majority] == target
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
                y_levels,
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
                y_levels,
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
    get_individual_classifiers(M::Model)

Returns the individual classifiers in the forest.
If the input is a decision tree, the method returns the decision tree itself inside an array.

# Arguments
- `M::TreeModel`: The model selected by the user.
- `model::CounterfactualExplanations.D`

# Returns
- `classifiers::AbstractArray`: An array of individual classifiers in the forest.
"""
function get_individual_classifiers(M::Model)
    fitted_params = MLJBase.fitted_params(M.model, M.fitresult)
    if M.model.model isa MLJDecisionTreeInterface.DecisionTreeClassifier
        return [fitted_params.tree.node]
    end
    trees = []
    for tree in fitted_params.forest.trees
        push!(trees, tree)
    end
    return trees
end