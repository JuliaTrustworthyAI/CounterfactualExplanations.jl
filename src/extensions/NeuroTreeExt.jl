"""
    NeuroTreeModel

Concrete type for differentiable tree-based models from `NeuroTreeModels`. Since `NeuroTreeModels` has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct NeuroTreeModel <: Models.MLJModelType end

function Models.Differentiability(::CounterfactualExplanations.NeuroTreeModel)
    return Models.IsDifferentiable()
end

Models.all_models_catalogue[:NeuroTreeModel] = CounterfactualExplanations.NeuroTreeModel
