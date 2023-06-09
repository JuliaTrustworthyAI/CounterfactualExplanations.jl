using DataFrames
using SliceMap
using EvoTrees
using MLJBase

"""
This type provides a basic interface to gradient-boosted tree models from the MLJ library.
However, this might not be the final version of the interface: full support for generating counterfactual explanations for EvoTrees has not been implemented yet.
"""

"""
    EvoTreeModel <: AbstractMLJModel

Constructor for gradient-boosted decision trees from the EvoTrees.jl library.

# Arguments
- `model::Any`: The model selected by the user. Must be a model from the MLJ library.
- `likelihood::Symbol`: The likelihood of the model. Must be one of `[:classification_binary, :classification_multi]`.

# Returns
- `EvoTreeModel`: An `EvoTreeClassifier` from `EvoTrees.jl` wrapped inside the EvoTreeModel class.
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
                    "`type` should be in `[:classification_binary, :classification_multi].
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
- `logits::Matrix`: A matrix of logits for each output class for each data point in X.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data point x
"""
function logits(M::EvoTreeModel, X::AbstractArray)
    p = probs(M, X)
    if M.likelihood == :classification_binary
        logits = log.(p ./ (1 .- p))
    else
        logits = log.(p)
    end
    return logits
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2})

Calculates the probability scores for each output class for the two-dimensional input data matrix X.

# Arguments
- `M::EvoTreeModel`: The EvoTree model.
- `X::AbstractArray`: The feature vector for which the predictions are made.

# Returns
- `p::Matrix`: A matrix of probability scores for each output class for each data point in X.

# Example
probabilities = Models.probs(M, X) # calculates the probability scores for each output class for each data point in X.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,2})
    output = MLJBase.predict(M.model, DataFrame(X', :auto))
    p = MLJBase.pdf(output, MLJBase.classes(output))'
    return p
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 1})

Works the same way as the probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 1-dimensional rather than 2-dimensional input data.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,1})
    X = reshape(X, 1, length(X))
    output = MLJBase.predict(M.model, DataFrame(X, :auto))
    p = MLJBase.pdf(output, MLJBase.classes(output))'
    return p
end

"""
    probs(M::EvoTreeModel, X::AbstractArray{<:Number, 3})

Works the same way as the probs(M::EvoTreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 3-dimensional rather than 2-dimensional input data.
"""
function probs(M::EvoTreeModel, X::AbstractArray{<:Number,3})
    # Slices the 3-dimensional input data into 1- and 2-dimensional arrays
    # and then calls the probs method for 1- and 2-dimensional input data on those slices
    output = SliceMap.slicemap(x -> probs(M, x), X; dims=[1, 2])
    p = MLJBase.pdf(output, MLJBase.classes(output))
    return p
end

"""
    EvoTreeModel(data::CounterfactualData; kwargs...)

Constructs a new EvoTreeModel object from the data in a `CounterfactualData` object.
Not called by the user directly.

# Arguments
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `model::EvoTreeModel`: The trained EvoTree model.
"""
function EvoTreeModel(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)

    model = EvoTrees.EvoTreeClassifier(kwargs...)
    mach = machine(model, X, y)

    return EvoTreeModel(mach, data.likelihood)
end

"""
    train(M::EvoTreeModel, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::EvoTreeModel`: The wrapper for an EvoTree model.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::EvoTreeModel`: The fitted EvoTree model.
"""
function train(M::EvoTreeModel, data::CounterfactualData; kwargs...)
    MLJBase.fit!(M.model)
    return M
end
