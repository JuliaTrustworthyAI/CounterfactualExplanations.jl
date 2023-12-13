"""
    map_from_latent(
        ce::CounterfactualExplanation,
        x::Union{AbstractArray,Nothing}=nothing,
    )

Maps the state variable back from the latent space to the feature space.
"""
function map_from_latent(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)

    # Unpack:
    s′ = isnothing(x) ? deepcopy(ce.s′) : x
    data = ce.data

    # Latent space:
    if ce.params[:latent_space]
        generative_model = data.generative_model
        if !isnothing(generative_model)
            # NOTE! This is not very clean, will be improved.
            if generative_model.params.nll == Flux.Losses.logitbinarycrossentropy
                s′ = Flux.σ.(generative_model.decoder(s′))
            else
                s′ = generative_model.decoder(s′)
            end
        end
    end

    return s′
end

@doc raw"""
   function map_to_latent(
        ce::CounterfactualExplanation,
        x::Union{AbstractArray,Nothing}=nothing,
    ) 

Maps `x` from the feature space $\mathcal{X}$ to the latent space learned by the generative model.
"""
function map_to_latent(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)

    # Unpack:
    s′ = isnothing(x) ? deepcopy(ce.s′) : x
    data = ce.data
    generator = ce.generator

    if ce.params[:latent_space]
        generative_model = DataPreprocessing.get_generative_model(
            data; ce.generative_model_params...
        )
        # map counterfactual to latent space: s′=z′∼p(z|x)
        s′, _, _ = GenerativeModels.rand(generative_model.encoder, s′)
    end

    return s′
end

"""
    wants_latent_space(
        ce::CounterfactualExplanation, 
        x::Union{AbstractArray,Nothing} = nothing,
    )   

A convenience function that checks if latent space search is applicable.
"""
function wants_latent_space(ce::CounterfactualExplanation)

    # Unpack:
    latent_space = ce.params[:latent_space]

    # If threshold is already reached, training GM is redundant:
    latent_space = latent_space && !Convergence.threshold_reached(ce)

    return latent_space
end
