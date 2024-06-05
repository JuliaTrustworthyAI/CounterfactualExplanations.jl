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
    function Model(model, likelihood, fitresult, type)
        @assert likelihood in [:classification_binary, :classification_multi] "Invalid likelihood specified. Currently supported likelihoods are `:classification_binary` and `:classification_multi`."
        return new(model, likelihood, fitresult, type)
    end
end

"""
    Model(type::AbstractModelType; likelihood::Symbol=:classification_binary)

Outer constructor for `Model` where the atomic model is not yet defined.
"""
function Model(type::AbstractModelType; likelihood::Symbol=:classification_binary)
    @assert likelihood in [:classification_binary, :classification_multi] "Invalid likelihood specified. Currently supported likelihoods are `:classification_binary` and `:classification_multi`."
    return Model(nothing, likelihood, nothing, type)
end

"""
    Model(model, type::AbstractModelType; likelihood::Symbol=:classification_binary)

Outer constructor for `Model` where the atomic model is defined and assumed to be pre-trained.
"""
function Model(model, type::AbstractModelType; likelihood::Symbol=:classification_binary)
    return Model(model, likelihood, model, type)
end

"""
    (model::AbstractModel)(X::AbstractArray)

When called on data `x`, logits are returned.
"""
(model::AbstractModel)(X::AbstractArray) = logits(model, X)

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
    (type::AbstractModelType)(data::CounterfactualData; kwargs...)

Wrap model `type` around the data in `data`. This is a convenience function to avoid having to construct a `Model` object.
"""
function (type::AbstractModelType)(data::CounterfactualData; kwargs...)
    return Model(type; likelihood=data.likelihood)(data; kwargs...)
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
- `type::AbstractModelType`: The type of model to be trained, e.g., `MLP`, `DecisionTreeModel`, etc.

# Examples

```jldoctest
julia> using CounterfactualExplanations

julia> using CounterfactualExplanations.Models

julia> using TaijaData

julia> data = CounterfactualData(load_linearly_separable()...);

julia> M = fit_model(data, Linear())
CounterfactualExplanations.Models.Model(Chain(Dense(2 => 2)), :classification_multi, Chain(Dense(2 => 2)), Linear())
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

"""
    fit_model(
        counterfactual_data::CounterfactualData, model::Symbol=:MLP;
        kwrgs...
    )

Fits one of the available default models to the `counterfactual_data`. The `model` argument can be used to specify the desired model. The available values correspond to the keys of the [`all_models_catalogue`](@ref) dictionary.
"""
function fit_model(counterfactual_data::CounterfactualData, model::Symbol=:MLP; kwrgs...)
    @assert model in keys(all_models_catalogue) "Specified model does not match any of the models available in the `all_models_catalogue`."
    type = all_models_catalogue[model]
    M = fit_model(counterfactual_data, type(); kwrgs...)
    return M
end
