using MLJDecisionTreeInterface
using DataFrames
using MLJBase

# The implementation of MLJ: DecisionTree: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl

"""
    TreeModel <: AbstractNonDifferentiableJuliaModel

Constructor for tree-based models from the MLJ library. 
"""
struct TreeModel <: AbstractNonDifferentiableJuliaModel
    mach::Any
    likelihood::Symbol
    function TreeModel(mach, likelihood)
        if !(mach.model isa MLJDecisionTreeInterface.DecisionTreeClassifier || mach.model isa MLJDecisionTreeInterface.RandomForestClassifier)
            throw(
                ArgumentError(
                    "model should be of type DecisionTreeClassifier or RandomForestClassifier",
                    ),
                )
        end
        if likelihood == :classification_binary
            new(mach, likelihood)
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

function TreeModel(data::CounterfactualData, likelihood::Symbol=:classification_binary)
    model = DecisionTreeClassifier()
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)

    X = Float32.(X')
    y = string.(y[2,:])

    DecisionTree.fit!(model, X, y)

    return TreeModel(model, likelihood)
end

"""
Outer constructor method for TreeModel.
"""
function TreeModel(mach::Any; likelihood::Symbol=:classification_binary)
    return TreeModel(mach, likelihood)
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
    if M.mach.model isa MLJDecisionTreeInterface.DecisionTreeClassifier
        return [M.mach.model]
    end
    return M.mach.model.trees
end

function logits(M::TreeModel, X::AbstractArray)
    df = DataFrame(reshape(X, 1, :), :auto)
    return MLJBase.predict(M.mach, df)
end

function probs(M::TreeModel, X::AbstractArray)
    df = DataFrame(reshape(X, 1, :), :auto)
    return pdf(MLJBase.predict(M.mach, df), MLJBase.report(M.mach).classes_seen)
end