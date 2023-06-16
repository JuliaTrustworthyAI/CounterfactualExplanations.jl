"""
    has_pretrained_generative_model(counterfactual_data::CounterfactualData)

Checks if generative model is present and trained.
"""
function has_pretrained_generative_model(counterfactual_data::CounterfactualData)
    return !isnothing(counterfactual_data.generative_model) &&
           counterfactual_data.generative_model.trained
end

"""
    get_generative_model(counterfactual_data::CounterfactualData)

Returns the underlying generative model. If there is no existing model available, the default generative model (VAE) is used. Otherwise it is expected that existing generative model has been pre-trained or else a warning is triggered.
"""
function get_generative_model(counterfactual_data::CounterfactualData; kwargs...)
    if !has_pretrained_generative_model(counterfactual_data)
        @info "No pre-trained generative model found. Using default generative model. Begin training."
        counterfactual_data.generative_model = GenerativeModels.VAE(
            input_dim(counterfactual_data); kwargs...
        )
        X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(counterfactual_data)
        # NOTE: y is not actually used, may refactor in the future to make that clearer.
        GenerativeModels.train!(counterfactual_data.generative_model, X, y)
        @info "Training of generative model completed."
    else
        if !counterfactual_data.generative_model.trained
            @warn "The provided generative model has not been trained. Latent space search is likely to perform poorly."
        end
    end
    return counterfactual_data.generative_model
end
