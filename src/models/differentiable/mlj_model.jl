using DataFrames
using SliceMap
using EvoTrees
using MLJBase

"""
This type provides a basic interface to gradient-boosted tree models from the MLJ library.
However, this is not be the final version of the interface: full support for EvoTrees has not been implemented yet and the `logits` and `probs` methods will be changed in the process of doing that if needed.
"""

"""
    EvoTreeModel <: AbstractMLJModel

Constructor for gradient-boosted decision trees from the EvoTrees.jl library. 
"""
struct EvoTreeModel <: AbstractDifferentiableModel
    model::Any
    likelihood::Symbol
    function EvoTreeModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary, :classification_multi]1.
                    Support for regressors has not been implemented yet.`"
                ),
            )
        end
    end
end

"""
Outer constructor method for EvoTreeModel.
"""
function EvoTreeModel(model::Any; likelihood::Symbol=:classification_binary)
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
    output = MLJBase.predict(M.model, DataFrame(X', :auto))
    p = pdf(output, classes(output))'
    return p
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 1})

Works the same way as the probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 1-dimensional rather than 2-dimensional input data.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,1})
    X = reshape(X, 1, length(X))
    output = MLJBase.predict(M.model, DataFrame(X, :auto))
    p = pdf(output, classes(output))'
    return p
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 3})

Works the same way as the probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 3-dimensional rather than 2-dimensional input data.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,3})
    output = SliceMap.slicemap(x -> probs(M,x), X, dims=[1,2])
    p = pdf(output, classes(output))
    return p
end


"""
The two methods below have been implemented only for testing purposes.
"""
function EvoTreeModel(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)

    X = Float32.(X)
    y = Float32.(y)[1, :]

    model = EvoTrees.EvoTreeClassifier()
    df_x = DataFrame(X', :auto)
    mach = machine(model, df_x, categorical(y))

    return EvoTreeModel(mach, data.likelihood)
end

function train(M::EvoTreeModel, data::CounterfactualData; kwargs...)
    MLJBase.fit!(M.model)
    return M
end
