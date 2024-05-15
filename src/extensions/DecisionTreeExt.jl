abstract type AbstractDecisionTree <: Models.MLJModelType end

"""
    DecisionTree

Concrete type for tree-based models from `DecisionTree`. Since `DecisionTree` has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct DecisionTree <: AbstractDecisionTree end

Models.all_models_catalogue[:DecisionTree] = CounterfactualExplanations.DecisionTree

"""
    RandomForest

Concrete type for tree-based models from `DecisionTree`. Since the `DecisionTree` package has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct RandomForest <: AbstractDecisionTree end

Models.all_models_catalogue[:RandomForest] = CounterfactualExplanations.RandomForest
