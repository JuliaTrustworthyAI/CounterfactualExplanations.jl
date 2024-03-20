using CounterfactualExplanations.Models

"""
    LaplaceReduxModel <: AbstractCustomDifferentiableModel

Constructor for `LaplaceRedux.jl` model.
"""
struct LaplaceReduxModel <: Models.AbstractCustomDifferentiableModel
    model::LaplaceRedux.Laplace
    likelihood::Symbol
    function LaplaceReduxModel(model, likelihood)
        @assert likelihood in [:classification_binary, :classification_multi] "Likelihood should be in `[:classification_binary, :classification_multi]`"
        return new(model, likelihood)
    end
end

# Outer constructor method:
function CounterfactualExplanations.LaplaceReduxModel(
    model; likelihood::Symbol=:classification_binary
)
    return LaplaceReduxModel(model, likelihood)
end

Models.logits(M::LaplaceReduxModel, X::AbstractArray) = M.model.model(X)
Models.probs(M::LaplaceReduxModel, X::AbstractArray) = LaplaceRedux.predict(M.model, X)
