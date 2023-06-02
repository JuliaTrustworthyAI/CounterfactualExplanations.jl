using MLJDecisionTreeInterface
using DataFrames
using MLJBase

# The implementation of MLJ: DecisionTree: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl

"""
    TreeModel <: AbstractNonDifferentiableJuliaModel

Constructor for tree-based models from the MLJ library. 
"""
struct TreeModel <: AbstractNonDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function TreeModel(model, likelihood)
        if !(model.model isa MLJDecisionTreeInterface.DecisionTreeClassifier || model.model isa MLJDecisionTreeInterface.RandomForestClassifier)
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
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary]`"
                ),
            )
        end
    end
end

"""
Outer constructor method for TreeModel.
"""
function TreeModel(model::Any; likelihood::Symbol=:classification_binary)
    return TreeModel(model, likelihood)
end

function TreeModel(data::CounterfactualData; likelihood::Symbol=:classification_binary)
    M = DecisionTreeClassifier()
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)

    X = Float32.(X)
    y = string.(y[2,:])

    df = DataFrame(X', :auto)
    model = machine(M, df, categorical(y)) |> MLJBase.fit!

    return TreeModel(model, likelihood)
end

# Methods
"""
    predict_label(M::TreeModel, X::AbstractArray)

Returns the predicted label for X.
"""
function predict_label(M::TreeModel, X::AbstractArray)
    return argmax(probs(M, X))[1]
end

"""
    get_individual_classifiers(M::TreeModel)

Returns the individual classifiers in the forest.
"""
function get_individual_classifiers(M::TreeModel)
    if M.model.model isa MLJDecisionTreeInterface.DecisionTreeClassifier
        return [M.model.model]
    end
    return M.model.model.trees
end

function logits(M::TreeModel, X::AbstractArray)
    p = probs(M, X)
    if M.likelihood == :classification_binary
        output = log.(p ./ (1 .- p))
    else
        output = log.(p)
    end
    return output
end

function probs(M::TreeModel, X::AbstractArray{<:Number, 2})
    output = MLJBase.predict(M.model, DataFrame(X', :auto))'
    if M.likelihood == :classification_binary
        output = reshape(output[2,:],1,size(output,2)) # binary case
    end
    return output
end

function probs(M::TreeModel, X::AbstractArray{<:Number, 1})
    X = reshape(X, 1,length(X))
    output = MLJBase.predict(M.model, DataFrame(X, :auto))'
    if M.likelihood == :classification_binary
        output = reshape(output[2,:],1,size(output,2)) # binary case
    end
    return output
end

# function probs(M::TreeModel, X::AbstractArray{<:Number, 3})
#     output = SliceMap.slicemap(x -> probs(M,x), X, dims=[1,2])
#     return output
# end