using MLJ: DecisionTree

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
                    "`type` should be in `[:classification_binary,:classification_multi]`"
                ),
            )
        end
    end
end


"""
Outer constructor method for TreeModel.
"""
function TreeModel(model; likelihood::Symbol=:classification_binary)
    return TreeModel(model, likelihood)
end


# Methods
function individual_outputs(M::TreeModel, X::AbstractArray)
    votes = [DecisionTree.apply_tree(tree, X) for tree in M.trees]
    return votes
end

predict(M::TreeModel, X::AbstractArray) = DecisionTree.apply_forest(M, X)
