"Type union for `DecisionTree` random forest classifiers and regressors."
const AtomicRandomForest = Union{
    MLJDecisionTreeInterface.RandomForestClassifier,
    MLJDecisionTreeInterface.RandomForestRegressor,
}

"""
    RandomForest(model::AtomicRandomForest; likelihood::Symbol=:classification_binary)

Outer constructor for random forests.
"""
function RandomForest(model::AtomicRandomForest; likelihood::Symbol=:classification_binary)
    return Models.Model(
        model, CounterfactualExplanations.RandomForest(); likelihood=likelihood
    )
end

"""
    (M::Models.Model)(
        data::CounterfactualData, type::CounterfactualExplanations.RandomForest; kwargs...
    )
    
Constructs a random forest for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData, type::CounterfactualExplanations.RandomForest; kwargs...
)
    model = MLJDecisionTreeInterface.RandomForestClassifier(; kwargs...)
    return RandomForest(model; likelihood=data.likelihood)
end
