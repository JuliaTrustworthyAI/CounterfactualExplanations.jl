"""
    NeuroTreeModel

Concrete type for differentiable tree-based models from `NeuroTreeModels`. Since `NeuroTreeModels` has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct NeuroTreeModel <: Models.MLJModelType end

Models.Differentiability(::CounterfactualExplanations.NeuroTreeModel) = Models.IsDifferentiable()

Models.all_models_catalogue[:NeuroTreeModel] = CounterfactualExplanations.NeuroTreeModel
