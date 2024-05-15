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

"Type union for `DecisionTree` random forest classifiers and regressors."
const AtomicRandomForest = Union{
    MLJDecisionTreeInterface.RandomForestClassifier, MLJDecisionTreeInterface.RandomForestRegressor
}

"""
    RandomForest(model::AtomicDecisionTree; likelihood::Symbol=:classification_binary)

Outer constructor for random forests.
"""
function RandomForest(model::AtomicDecisionTree; likelihood::Symbol=:classification_binary)
    return Models.Model(
        model,
        CounterfactualExplanations.RandomForest();
        likelihood=likelihood,
    )
end