using MLJDecisionTreeInterface
using DataFrames
using MLJBase

# The implementation of MLJ: DecisionTree: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl

"""
    TreeModel <: AbstractNonDifferentiableJuliaModel

Constructor for tree-based models from the MLJ library. 

# Arguments
- `model::Any`: The model selected by the user. Must be a model from the MLJ library.
- `likelihood::Symbol`: The likelihood of the model. Must be one of `[:classification_binary, :classification_multi]`.

# Returns
- `TreeModel`: A tree-based model from the MLJ library wrapped inside the TreeModel class.
"""
struct TreeModel <: AbstractNonDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function TreeModel(model, likelihood)
        if !(
            model.model isa MLJDecisionTreeInterface.DecisionTreeClassifier ||
            model.model isa MLJDecisionTreeInterface.RandomForestClassifier
        )
            throw(
                ArgumentError(
                    "model should be of type DecisionTreeClassifier or RandomForestClassifier",
                ),
            )
        end
        if likelihood == :classification_binary
            new(model, likelihood)
        elseif likelihood == :classification_multi
            throw(
                ArgumentError(
                    "`type` should be `:classification_binary`. Support for multi-class classification with tree-based models is not yet implemented.",
                ),
            )
        else
            throw(ArgumentError("`type` should be in `[:classification_binary]`"))
        end
    end
end

"""
Outer constructor method for TreeModel.
"""
function TreeModel(model::Any; likelihood::Symbol=:classification_binary)
    return TreeModel(model, likelihood)
end

# Methods
"""
    predict_label(M::TreeModel, X::AbstractArray)

Returns the predicted label for `X`.

# Arguments
- `M::TreeModel`: The model selected by the user.
- `X::AbstractArray`: The input array for which the label is predicted.

# Returns
- `labels::AbstractArray`: The predicted label for each data point in `X`.

# Example
label = Models.predict_label(M, x) # returns the predicted label for each data point in `x`
"""
function predict_label(M::TreeModel, X::AbstractArray)
    return MLJBase.predict_mode(M.model, DataFrame(X', :auto))
end

"""
    get_individual_classifiers(M::TreeModel)

Returns the individual classifiers in the forest.
If the input is a decision tree, the method returns the decision tree itself inside an array.

# Arguments
- `M::TreeModel`: The model selected by the user.

# Returns
- `classifiers::AbstractArray`: An array of individual classifiers in the forest.

# Example
classifiers = Models.get_individual_classifiers(M) # returns the individual classifiers in the forest
"""
function get_individual_classifiers(M::TreeModel)
    if M.model.model isa MLJDecisionTreeInterface.DecisionTreeClassifier
        return [M]
    end
    trees = []
    for tree in M.model.trees
        push!(trees, TreeModel(tree, :classification_binary))
    end
    return trees
end

"""
    logits(M::TreeModel, X::AbstractArray)

Calculates the logit scores output by the model M for the input data X.

# Arguments
- `M::TreeModel`: The model selected by the user.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::Matrix`: A matrix of logits for each output class for each data point in X.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data point x
"""
function logits(M::TreeModel, X::AbstractArray)
    p = probs(M, X)
    if M.likelihood == :classification_binary
        output = log.(p ./ (1 .- p))
    else
        output = log.(p)
    end
    return output
end

"""
    probs(M::TreeModel, X::AbstractArray{<:Number, 2})

Calculates the probability scores for each output class for the two-dimensional input data matrix X.

# Arguments
- `M::TreeModel`: The TreeModel.
- `X::AbstractArray`: The feature vector for which the predictions are made.

# Returns
- `p::Matrix`: A matrix of probability scores for each output class for each data point in X.

# Example
probabilities = Models.probs(M, X) # calculates the probability scores for each output class for each data point in X.
"""
function probs(M::TreeModel, X::AbstractArray{<:Number,2})
    output = MLJBase.predict(M.model, DataFrame(X', :auto))
    p = pdf(output, classes(output))'
    if M.likelihood == :classification_binary
        p = reshape(p[2, :], 1, size(p, 2))
    end
    return p
end

"""
    probs(M::TreeModel, X::AbstractArray{<:Number, 1})

Works the same way as the probs(M::TreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 1-dimensional rather than 2-dimensional input data.
"""
function probs(M::TreeModel, X::AbstractArray{<:Number,1})
    X = reshape(X, 1, length(X))
    output = MLJBase.predict(M.model, DataFrame(X, :auto))
    p = pdf(output, classes(output))'
    if M.likelihood == :classification_binary
        p = reshape(p[2, :], 1, size(p, 2))
    end
    return p
end

"""
    probs(M::TreeModel, X::AbstractArray{<:Number, 3})

Works the same way as the probs(M::TreeModel, X::AbstractArray{<:Number, 2}) method above, but handles 3-dimensional rather than 2-dimensional input data.
"""
function probs(M::TreeModel, X::AbstractArray{<:Number,3})
    # Slices the 3-dimensional input data into 1- and 2-dimensional arrays
    # and then calls the probs method for 1- and 2-dimensional input data on those slices
    output = SliceMap.slicemap(x -> probs(M, x), X; dims=[1, 2])
    p = pdf(output, classes(output))
    return p
end

"""
    DecisionTreeModel(data::CounterfactualData; kwargs...)

Constructs a new TreeModel object wrapped around a decision tree from the data in a `CounterfactualData` object.
Not called by the user directly.

# Arguments
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `model::TreeModel`: A TreeModel object.
"""
function DecisionTreeModel(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)

    M = MLJDecisionTreeInterface.DecisionTreeClassifier(kwargs...)
    model = MLJBase.machine(M, X, y)

    return TreeModel(model, :classification_binary)
end

"""
    RandomForestModel(data::CounterfactualData; kwargs...)

Constructs a new TreeModel object wrapped around a random forest from the data in a `CounterfactualData` object.
Not called by the user directly.

# Arguments
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `model::TreeModel`: A TreeModel object.
"""
function RandomForestModel(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)

    M = MLJDecisionTreeInterface.DecisionTreeClassifier(kwargs...)
    model = MLJBase.machine(M, X, y)

    return TreeModel(model, :classification_binary)
end

"""
    train(M::TreeModel, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::TreeModel`: The wrapper for a TreeModel.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::TreeModel`: The fitted TreeModel.
"""
function train(M::TreeModel, data::CounterfactualData; kwargs...)
    MLJBase.fit!(M.model)
    return M
end
