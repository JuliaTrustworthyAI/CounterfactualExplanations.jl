"Type union for `DecisionTree` random forest classifiers and regressors."
const AtomicRandomForest = Union{
    MLJDecisionTreeInterface.RandomForestClassifier,
    MLJDecisionTreeInterface.RandomForestRegressor,
}

"""
    (M::Models.Model)(
        data::CounterfactualData, type::CounterfactualExplanations.RandomForestModel; kwargs...
    )
    
Constructs a random forest for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData, type::CounterfactualExplanations.RandomForestModel; kwargs...
)
    model = MLJDecisionTreeInterface.RandomForestClassifier(; kwargs...)
    return CounterfactualExplanations.RandomForestModel(model; likelihood=data.likelihood)
end
