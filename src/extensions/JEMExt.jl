"""
    JEM

Concrete type for joint-energy models from `JointEnergyModels`. Since `JointEnergyModels` has an MLJ interface, we subtype the `MLJModelType` model type.
"""
struct JEM <: Models.MLJModelType end

function Models.Differentiability(::CounterfactualExplanations.JEM)
    return Models.IsDifferentiable()
end

Models.all_models_catalogue[:JEM] = CounterfactualExplanations.JEM
