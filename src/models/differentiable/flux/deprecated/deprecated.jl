include("mlp.jl")
include("ensemble.jl")

"""
    fit_model(
        counterfactual_data::CounterfactualData, model::Symbol=:MLP;
        kwrgs...
    )

Fits one of the available default models to the `counterfactual_data`. The `model` argument can be used to specify the desired model. The available values correspond to the keys of the [`all_models_catalogue`](@ref) dictionary.
"""
function fit_model(counterfactual_data::CounterfactualData, model::Symbol=:MLP; kwrgs...)
    @assert model in keys(all_models_catalogue) "Specified model does not match any of the models available in the `all_models_catalogue`."

    # Set up:
    M = all_models_catalogue[model](counterfactual_data; kwrgs...)
    M = train(M, counterfactual_data)

    return M
end
