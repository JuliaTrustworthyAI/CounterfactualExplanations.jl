"""
    LaplaceReduxModel <: AbstractCustomDifferentiableModel

Constructor for `LaplaceRedux.jl` model.
"""
struct LaplaceReduxModel <: AbstractCustomDifferentiableModel
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

# Methods
# The methods are commented out for the time being, as they are not called from anywhere in the package
# logits(M::LaplaceReduxModel, X::AbstractArray) = M.model.model(X)
# probs(M::LaplaceReduxModel, X::AbstractArray) = LaplaceRedux.predict(M.model, X)
