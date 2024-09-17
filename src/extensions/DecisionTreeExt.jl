abstract type AbstractDecisionTree <: Models.MLJModelType end

"""
    DecisionTreeModel

Concrete type for tree-based models from `DecisionTree.jl`. Since `DecisionTree.jl` has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct DecisionTreeModel <: AbstractDecisionTree end

Models.all_models_catalogue[:DecisionTreeModel] =
    CounterfactualExplanations.DecisionTreeModel

"""
    RandomForestModel

Concrete type for random forest model from `DecisionTree.jl`. Since the `DecisionTree` package has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct RandomForestModel <: AbstractDecisionTree end

Models.all_models_catalogue[:RandomForestModel] =
    CounterfactualExplanations.RandomForestModel
