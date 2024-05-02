abstract type AbstractModelType end

"""
    Model <: AbstractModel

Constructor for all models.
"""
mutable struct Model <: AbstractModel
    model
    likelihood::Symbol
    fitresult
    type::AbstractModelType
end

"""
    Model(type::AbstractModelType; likelihood::Symbol=:classification_binary)

Outer constructor for `Model` where the atomic model is not yet defined.
"""
function Model(type::AbstractModelType; likelihood::Symbol=:classification_binary)
    return Model(nothing, likelihood, nothing, type)
end

"""
    Model(model, type::AbstractModelType; likelihood::Symbol=:classification_binary)

Outer constructor for `Model` where the atomic model is defined, but the model has not been trained yet.
"""
function Model(model, type::AbstractModelType; likelihood::Symbol=:classification_binary)
    return Model(model, likelihood, nothing, type)
end

"""
    logits(M::Model, X::AbstractArray)

Returns the logits of the model.
"""
logits(M::Model, X::AbstractArray) = logits(M, M.type, X)

"""
    probs(M::Model, X::AbstractArray)

Returns the probabilities of the model.
"""
probs(M::Model, X::AbstractArray) = probs(M, M.type, X)

"""
    (M::Model)(data::CounterfactualData; kwargs...)

Wrap model `M` around the data in `data`.
"""
function (M::Model)(data::CounterfactualData; kwargs...)
    return (M::Model)(data, M.type; kwargs...)
end

"""
    train(M::Model, data::CounterfactualData)

Trains the model `M` on the data in `data`.
"""
function train(M::Model, data::CounterfactualData)
    return train(M, M.type, data)
end

"""
    fit_model(
        counterfactual_data::CounterfactualData, type::AbstractModelType; kwrgs...
    )

A wrapper function to fit a model to the `counterfactual_data` for a given `type` of model.

# Arguments

- `counterfactual_data::CounterfactualData`: The data to be used for training the model.
- `type::AbstractModelType`: The type of model to be trained, e.g., `MLP`, `DecisionTree`, etc.

# Examples

```jldoctest
julia> using CounterfactualExplanations

julia> using CounterfactualExplanations.Models

julia> using TaijaData

julia> data = CounterfactualData(load_linearly_separable()...);

julia> M = fit_model(data, Linear())
Model(Chain(Dense(2 => 2)), :classification_multi, nothing, Linear())
```
"""
function fit_model(
    counterfactual_data::CounterfactualData, type::AbstractModelType; kwrgs...
)
    M =
        Model(type; likelihood=counterfactual_data.likelihood)(
            counterfactual_data; kwrgs...
        ) |> M -> train(M, counterfactual_data)

    return M
end
