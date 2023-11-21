using CounterfactualExplanations.Models

"""
    LaplaceReduxModel <: AbstractCustomDifferentiableModel

Constructor for `LaplaceRedux.jl` model.
"""
struct LaplaceReduxModel <: Models.AbstractCustomDifferentiableModel
    model::LaplaceRedux.Laplace
    likelihood::Symbol
    function LaplaceReduxModel(model, likelihood)
        if likelihood == :classification_binary
            new(model, likelihood)
        elseif likelihood == :classification_multi
            throw(
                ArgumentError(
                    "`type` should be `:classification_binary`. Support for multi-class Laplace Redux is not yet implemented.",
                ),
            )
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi]`"
                ),
            )
        end
    end
end

# Outer constructor method:
function LaplaceReduxModel(model; likelihood::Symbol=:classification_binary)
    return LaplaceReduxModel(model, likelihood)
end

Models.logits(M::LaplaceReduxModel, X::AbstractArray) = M.model.model(X)
Models.probs(M::LaplaceReduxModel, X::AbstractArray) = LaplaceRedux.predict(M.model, X)
