"""
    NeuroTree

Concrete type for differentiable tree-based models from `NeuroTreeModels`. Since `NeuroTreeModels` has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct NeuroTree <: Models.MLJModelType end

Models.all_models_catalogue[:NeuroTree] = CounterfactualExplanations.NeuroTree
