using CounterfactualExplanations.Models

"Type union for `DecisionTree` decision tree classifiers and regressors."
const AtomicDecisionTree = Union{
    MLJDecisionTreeInterface.DecisionTreeClassifier, MLJDecisionTreeInterface.DecisionTreeRegressor
}

"""
    DecisionTree(model::AtomicDecisionTree; likelihood::Symbol=:classification_binary)

Outer constructor for a decision trees.
"""
function DecisionTree(model::AtomicDecisionTree; likelihood::Symbol=:classification_binary)
    return Models.Model(
        model,
        CounterfactualExplanations.DecisionTree();
        likelihood=likelihood,
    )
end

"""
    (M::Models.Model)(
        data::CounterfactualData,
        type::CounterfactualExplanations.DecisionTree;
        kwargs...,
    )
    
Constructs a decision tree for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData,
    type::CounterfactualExplanations.DecisionTree;
    kwargs...,
)
    model = MLJDecisionTreeInterface.DecisionTreeClassifier(; kwargs...)
    return DecisionTree(model; likelihood=data.likelihood)
end