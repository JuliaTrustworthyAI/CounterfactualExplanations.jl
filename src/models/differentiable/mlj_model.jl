using DataFrames: DataFrames
using SliceMap: SliceMap
using EvoTrees: EvoTrees

"""
This type provides a basic interface to differentiable models from the MLJ library.
However, this is not be the final version of the interface: full support for EvoTrees has not been implemented yet and the `logits` and `probs` methods will be changed in the process of doing that if needed.
"""

"""
    EvoTreeModel <: AbstractMLJModel

Constructor for gradient-boosted decision trees from the EvoTrees.jl library. 
"""
struct EvoTreeModel <: AbstractDifferentiableModel
    model::EvoTrees.GBTree
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
function EvoTreeModel(model::EvoTrees.GBTree; likelihood::Symbol=:classification_binary)
    return EvoTreeModel(model, likelihood)
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
function logits(M::EvoTreeModel, X::AbstractArray)
    p = probs(M, X)
    if M.likelihood == :classification_binary
        output = log.(p ./ (1 .- p))
    else
        output = log.(p)
    end
    return output
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2})

Calculates the probability scores for each output class for the two-dimensional input data matrix X.

# Arguments
- `M::EvoTreeModel`: The EvoTree model.
- `X::AbstractArray`: The feature vector for which the predictions are made.

# Returns
- `counterfactual_data::CounterfactualData`: A `CounterfactualData` object containing the preprocessed data.

# Example
probabilities = Models.probs(M, X) # calculates the probability scores for each output class for each data point in X.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,2})
    output = EvoTrees.predict(M.model, X')'
    if M.likelihood == :classification_binary
        output = reshape(output[2, :], 1, size(output, 2))
    end
    return output
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 1})

Works the same way as the probs(M::MLJModel, X::AbstractArray{<:Number, 2}) method above, but handles 1-dimensional rather than 2-dimensional input data.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,1})
    X = reshape(X, 1, length(X))
    output = EvoTrees.predict(M.model, X)'
    if M.likelihood == :classification_binary
        output = reshape(output[2, :], 1, size(output, 2))
    end
    return output
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 3})

Works the same way as the probs(M::MLJModel, X::AbstractArray{<:Number, 2}) method above, but handles 3-dimensional rather than 2-dimensional input data.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,3})
    output = SliceMap.slicemap(x -> probs(M, x), X; dims=[1, 2])
    return output
end