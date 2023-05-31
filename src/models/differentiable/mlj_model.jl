"""
This type provides a basic interface to differentiable models from the MLJ library.
The interface is currently incomplete and will be modified in the future as we add support for specific models from the MLJ registry.
"""

"""
    MLJModel <: AbstractDifferentiableModel

Constructor for differentiable models from the MLJ library. 
"""
struct MLJModel <: AbstractDifferentiableModel
    model::Any
    likelihood::Symbol
    function MLJModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi].
                    Support for regressors has not been implemented yet.`"
                ),
            )
        end
    end
end

"""
Outer constructor method for MLJModel.
"""
function MLJModel(model::Any; likelihood::Symbol=:classification_binary)
    return MLJModel(model, likelihood)
end