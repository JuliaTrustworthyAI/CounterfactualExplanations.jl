"Type union for `DecisionTree` random forest classifiers and regressors."
const AtomicRandomForest = Union{
    MLJDecisionTreeInterface.RandomForestClassifier,
    MLJDecisionTreeInterface.RandomForestRegressor,
}

"""
    CounterfactualExplanations.RandomForestModel(
        model::AtomicRandomForest; likelihood::Symbol=:classification_binary
    )

Outer constructor for random forests.
"""
function CounterfactualExplanations.RandomForestModel(
    model::AtomicRandomForest;
    fitresult=nothing,
    likelihood::Symbol=:classification_binary,
)
    return Models.Model(
        model,
        CounterfactualExplanations.RandomForestModel();
        fitresult=fitresult,
        likelihood=likelihood,
    )
end

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
