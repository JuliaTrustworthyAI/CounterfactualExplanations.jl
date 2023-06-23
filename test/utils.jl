using CounterfactualExplanations.Data
using CounterfactualExplanations.Models

function _load_synthetic()
    # Data:
    data_sets = Dict(
        :classification_binary => load_linearly_separable(),
        :classification_multi => load_multi_class(),
    )
    # Models
    synthetic = Dict()
    for (likelihood, data) in data_sets
        models = Dict()
        for (model_name, model) in standard_models_catalogue
            M = fit_model(data, model_name)
            models[model_name] = Dict(:raw_model => M.model, :model => M)
        end
        synthetic[likelihood] = Dict(:models => models, :data => data)
    end
    return synthetic
end

function get_target(counterfactual_data::CounterfactualData, factual_label::RawTargetType)
    target = rand(
        counterfactual_data.y_levels[counterfactual_data.y_levels .!= factual_label]
    )
    return target
end

"""
    _load_pretrained_models()

Loads pretrained Flux models.
"""
function _load_pretrained_models()
    pretrained = Dict(
        :cifar_10 => Dict(
            :models => Dict(            
                :mlp => Models.load_cifar_10_mlp(),
                :ensemble => Models.load_cifar_10_ensemble(),
            ),
            :latent => Dict(
                :vae_strong => Models.load_cifar_10_vae(strong=true),
                :vae_weak => Models.load_cifar_10_vae(strong=false),
            ),
        ),
        :mnist => Dict(
            :models => Dict(
                :mlp => Models.load_mnist_mlp(),
                :ensemble => Models.load_mnist_ensemble(),
            ),
            :latent => Dict(
                :vae_strong => Models.load_mnist_vae(strong=true),
                :vae_weak => Models.load_mnist_vae(strong=false),
            ),
        ),
        :fashion_mnist => Dict(
            :models => Dict(
                :mlp => Models.load_fashion_mnist_mlp(),
                :ensemble => Models.load_fashion_mnist_ensemble(),
            ),
            :latent => Dict(
                :vae_strong => Models.load_fashion_mnist_vae(strong=true),
                :vae_weak => Models.load_fashion_mnist_vae(strong=false),
            ),
        ),
    )
    return pretrained
end
