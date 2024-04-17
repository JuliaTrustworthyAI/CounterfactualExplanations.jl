using Flux: Flux

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
    if ce.generator.latent_space
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

    if generator.latent_space
        generative_model = DataPreprocessing.get_generative_model(
            data; generator.generative_model_params...
        )
        # map counterfactual to latent space: s′=z′∼p(z|x)
        s′, _, _ = GenerativeModels.rand(generative_model.encoder, s′)
    end

    return s′
end
