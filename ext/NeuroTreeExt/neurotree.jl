using CounterfactualExplanations.Models
using Flux
using MLJBase
using Tables: columntable

const MLJNeuroTreeModel = Union{
    NeuroTreeModels.NeuroTreeClassifier,NeuroTreeModels.NeuroTreeRegressor
}

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
    model::MLJNeuroTreeModel
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
function NeuroTreeModels.NeuroTreeModel(
    model::MLJNeuroTreeModel; likelihood::Symbol=:classification_binary, fitresult
)
    return CounterfactualExplanations.Models.NeuroTreeModel(model, likelihood, fitresult)
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
function NeuroTreeModels.NeuroTreeModel(data::CounterfactualData; kwargs...)
    outsize = length(data.y_levels)
    model = NeuroTreeModels.NeuroTreeClassifier(; outsize=outsize, kwargs...)
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
function Models.train(M::NeuroTreeModel, data::CounterfactualData)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    if M.likelihood ∉ [:classification_multi, :classification_binary]
        y = float.(y.refs)
    end
    X = columntable(X)
    mach = MLJBase.machine(M.model, X, y)
    MLJBase.fit!(mach)
    Flux.testmode!(mach.fitresult.chain)
    M = NeuroTreeModel(M.model, M.likelihood, mach.fitresult)
    return M
end

"""
    Models.logits(M::NeuroTreeModel, X::AbstractArray)

Calculates the logit scores output by the model M for the input data X.

# Arguments
- `M::NeuroTreeModel`: The model selected by the user. Must be a model from the MLJ library.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::Matrix`: A matrix of logits for each output class for each data point in X.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data point x
"""
Models.logits(M::NeuroTreeModel, X::AbstractArray) = M.fitresult(X)

"""
    Models.probs(M::NeuroTreeModel, X::AbstractArray{<:Number, 2})

Calculates the probability scores for each output class for the two-dimensional input data matrix X.

# Arguments
- `M::NeuroTreeModel`: The NeuroTree model.
- `X::AbstractArray`: The feature vector for which the predictions are made.

# Returns
- `p::Matrix`: A matrix of probability scores for each output class for each data point in X.

# Example
probabilities = Models.probs(M, X) # calculates the probability scores for each output class for each data point in X.
"""
Models.probs(M::NeuroTreeModel, X::AbstractArray{<:Number,2}) = softmax(logits(M, X))

"""
    Models.probs(M::NeuroTreeModel, X::AbstractArray{<:Number, 1})

Works the same way as the probs(M::NeuroTreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 1-dimensional rather than 2-dimensional input data.
"""
function Models.probs(M::NeuroTreeModel, X::AbstractArray{<:Number,1})
    X = reshape(X, 1, length(X))
    return Models.probs(M, X)
end
