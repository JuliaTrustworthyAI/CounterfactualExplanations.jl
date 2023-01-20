using CounterfactualExplanations.Data
using CounterfactualExplanations.Models

function _load_synthetic()
    # Data:
    data_sets = Dict(
        :classification_binary => load_linearly_separable(),
        :classification_multi => load_multi_class()
    )
    # Models
    synthetic = Dict()
    for (likelihood, data) in data_sets
        models = Dict()
        for (model_name, model) in model_catalogue
            M = fit_model(data, model_name)
            models[model_name] = Dict(
                :raw_model => M.model,
                :model => M
            )
        end
        synthetic[likelihood] = Dict(
            :models => models, 
            :data => data,
        )
    end
    return synthetic
end