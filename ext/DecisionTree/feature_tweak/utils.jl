using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models

"""
    feature_tweaking!(ce::AbstractCounterfactualExplanation)

Returns a counterfactual instance of `ce.x` based on the ensemble of classifiers provided.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object.

# Returns
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object.

# Example
ce = feature_tweaking!(ce) # returns a counterfactual inside the ce.s′ field based on the ensemble of classifiers provided
"""
function feature_tweaking!(ce::AbstractCounterfactualExplanation)
    @assert isa(ce.generator, Generators.FeatureTweakGenerator) "The feature tweak algorithm can only be applied using the feature tweak generator"
    @assert isa(ce.M.type, CounterfactualExplanations.AbstractDecisionTree) "The `FeatureTweakGenerator` currently only supports tree models. The counterfactual search will be terminated."

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