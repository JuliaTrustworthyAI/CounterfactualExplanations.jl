using CounterfactualExplanations.Models

"Type union for `DecisionTree` decision tree classifiers and regressors."
const AtomicDecisionTree = Union{
    MLJDecisionTreeInterface.DecisionTreeClassifier,
    MLJDecisionTreeInterface.DecisionTreeRegressor,
}

"""
    CounterfactualExplanations.DecisionTreeModel(
        model::AtomicDecisionTree; likelihood::Symbol=:classification_binary
    )

Outer constructor for a decision trees.
"""
function CounterfactualExplanations.DecisionTreeModel(
    model::AtomicDecisionTree; likelihood::Symbol=:classification_binary
)
    return Models.Model(
        model, CounterfactualExplanations.DecisionTreeModel(); likelihood=likelihood
    )
end

"""
    (M::Models.Model)(
        data::CounterfactualData,
        type::CounterfactualExplanations.DecisionTreeModel;
        kwargs...,
    )
    
Constructs a decision tree for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData, type::CounterfactualExplanations.DecisionTreeModel; kwargs...
)
    model = MLJDecisionTreeInterface.DecisionTreeClassifier(; kwargs...)
    return CounterfactualExplanations.DecisionTreeModel(model; likelihood=data.likelihood)
end
