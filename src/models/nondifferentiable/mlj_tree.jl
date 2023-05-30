using DecisionTree

# The implementation of MLJ: DecisionTree: https://github.com/JuliaAI/DecisionTree.jl/blob/dev/src/classification/main.jl

"""
    TreeModel <: AbstractNonDifferentiableJuliaModel

Constructor for tree-based models from the MLJ library. 
"""
struct TreeModel <: AbstractNonDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function TreeModel(model, likelihood)
        if !(model isa DecisionTreeClassifier || model isa RandomForestClassifier)
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
function TreeModel(model::Any; likelihood::Symbol=:classification_binary)
    return TreeModel(model, likelihood)
end


# Methods
"""
    predict_label(M::TreeModel, input_data::CounterfactualData, X::AbstractArray)

Returns the predicted label for X.
"""
function predict_label(M::TreeModel, X::AbstractArray)
    if M.model isa DecisionTreeClassifier
        return DecisionTree.predict(M.model, X)
    end
    return DecisionTree.predict(M.model, X)
end


"""
    get_individual_classifiers(M::TreeModel)

Returns the individual classifiers in the forest.
"""
function get_individual_classifiers(M::TreeModel)
    if M.model isa DecisionTreeClassifier
        return [M.model]
    end
    return M.model.trees
end


function logits(M::TreeModel, X::AbstractArray)
    return DecisionTree.predict_proba(M.model, X)
end


function probs(M::TreeModel, X::AbstractArray)
    return DecisionTree.predict_proba(M.model, X)
end
