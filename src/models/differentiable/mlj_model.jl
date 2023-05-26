using DataFrames
using MLJBase

"""
    MLJModel <: AbstractDifferentiableModel

Constructor for differentiable models from the MLJ library. 
"""
struct MLJModel <: AbstractDifferentiableModel
    mach::Any
    likelihood::Symbol
    function MLJModel(mach, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(mach, likelihood)
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
function MLJModel(mach::Any; likelihood::Symbol=:classification_binary)
    return MLJModel(mach, likelihood)
end


# Methods
"""
    logits(M::MLJModel, X::AbstractArray)

Calculates the logit scores output by the model M for the input data X.

# Arguments
- `M::MLJModel`: The model selected by the user. Must be a model from the MLJ library.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed data.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data point x
"""
function logits(M::MLJModel, X::AbstractArray)
    p = probs(M, X)
    if M.likelihood == :classification_binary
        output = log.(p./(1 .- p))
    else
        output = log.(p)
    end
    return output
end

"""
    probs(M::MLJModel, X::AbstractArray)

Calculates the probability scores for each output class for the input data X.

# Arguments
- `M::MLJModel`: The model selected by the user. Must be a model from the MLJ library.
- `X::AbstractArray`: The feature vector for which the predictions are made.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed data.

# Example
probabilities = Models.probs(M, x) # calculates the probability scores for each output class for the data point x
"""
function probs(M::MLJModel, X::AbstractArray)
    df = DataFrame(reshape(X, 1, :), :auto)
    prediction = MLJBase.predict(M.mach, df)
    probs_array = [pdf(pred, level) for pred in prediction for level in levels(pred)]
    return probs_array
end