"""
    _load_synthetic()

Loads synthetic data, models, and generators.
"""
function _load_synthetic()
    linearly_separable = TaijaData.load_linearly_separable()
    multi_class = TaijaData.load_multi_class()

    # Data:
    data_sets = Dict(
        :classification_binary => CounterfactualExplanations.DataPreprocessing.CounterfactualData(
            linearly_separable[1], linearly_separable[2]
        ),
        :classification_multi => CounterfactualExplanations.DataPreprocessing.CounterfactualData(
            multi_class[1], multi_class[2]
        ),
    )
    # Models
    synthetic = Dict()
    for (likelihood, data) in data_sets
        models = Dict()
        for (model_name, model) in Models.standard_models_catalogue
            M = fit_model(data, model_name)
            models[model_name] = Dict(:raw_model => M.model, :model => M)
        end
        synthetic[likelihood] = Dict(:models => models, :data => data)
    end
    return synthetic
end

"""
    get_target(counterfactual_data::CounterfactualData, factual_label::RawTargetType)

Returns a target label that is different from the factual label.
"""
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
        :mnist => Dict(
            :models => Dict(
                :mlp => Models.load_mnist_mlp(),
                :ensemble => Models.load_mnist_ensemble(),
            ),
            :latent => Dict(
                :vae_strong => Models.load_mnist_vae(; strong=true),
                :vae_weak => Models.load_mnist_vae(; strong=false),
            ),
        ),
        :fashion_mnist => Dict(
            :models => Dict(
                :mlp => Models.load_fashion_mnist_mlp(),
                :ensemble => Models.load_fashion_mnist_ensemble(),
            ),
            :latent => Dict(
                :vae_strong => Models.load_fashion_mnist_vae(; strong=true),
                :vae_weak => Models.load_fashion_mnist_vae(; strong=false),
            ),
        ),
    )

    if !Sys.iswindows()
        pretrained[:cifar_10] = Dict(
            :models => Dict(
                :mlp => Models.load_cifar_10_mlp(),
                :ensemble => Models.load_cifar_10_ensemble(),
            ),
            :latent => Dict(
                :vae_strong => Models.load_cifar_10_vae(; strong=true),
                :vae_weak => Models.load_cifar_10_vae(; strong=false),
            ),
        )
    end

    return pretrained
end
