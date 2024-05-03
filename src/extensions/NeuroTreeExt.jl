"""
    NeuroTree

Concrete type for differentiable tree-based models from `NeuroTreeModels`.
"""
struct NeuroTree <: Models.MLJModelType end

Models.all_models_catalogue[:NeuroTree] = CounterfactualExplanations.NeuroTree