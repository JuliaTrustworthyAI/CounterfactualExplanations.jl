using CounterfactualExplanations.Models
using MLJBase

"""
    NeuroTreeModel <: AbstractMLJModel

Constructor for gradient-boosted decision trees from the NeuroTrees.jl library.

# Arguments
- `model::Any`: The model selected by the user. Must be a model from the MLJ library.
- `likelihood::Symbol`: The likelihood of the model. Must be one of `[:classification_binary, :classification_multi]`.

# Returns
- `NeuroTreeModel`: An `NeuroTreeRegressor` from `NeuroTreeModels.jl` wrapped inside the NeuroTreeModel class.
"""
struct NeuroTreeModel <: Models.AbstractMLJModel
    model::NeuroTreeModels.NeuroTreeRegressor
    likelihood::Symbol
    fitresult::Any
    function NeuroTreeModel(model, likelihood, fitresult)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood, fitresult)
        else
            throw(
                ArgumentError(
                    "`likelihood` should be in `[:classification_binary, :classification_multi].
                    Support for regressors has not been implemented yet.`",
                ),
            )
        end
    end
end

"""
Outer constructor method for NeuroTreeModel.
"""
function CounterfactualExplanations.NeuroTreeModel(
    model; likelihood::Symbol=:classification_binary
)
    return NeuroTreeModel(model, likelihood, nothing)
end

"""
    NeuroTreeModel(data::CounterfactualData; kwargs...)

Constructs a new NeuroTreeModel object from the data in a `CounterfactualData` object.
Not called by the user directly.

# Arguments
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `model::NeuroTreeModel`: The NeuroTree model.
"""
function CounterfactualExplanations.NeuroTreeModel(data::CounterfactualData; kwargs...)
    model = NeuroTreeModels.NeuroTreeRegressor(; loss=:mlogloss, kwargs...)
    return NeuroTreeModel(model, data.likelihood, nothing)
end

"""
    train(M::NeuroTreeModel, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::NeuroTreeModel`: The wrapper for an NeuroTree model.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::NeuroTreeModel`: The fitted NeuroTree model.
"""
function Models.train(M::NeuroTreeModel, data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    mach = MLJBase.machine(M.model, X, y)
    MLJBase.fit!(mach)
    M.fitresult = mach.fitresult
    return M
end
