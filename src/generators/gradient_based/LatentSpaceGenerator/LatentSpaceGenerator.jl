mutable struct LatentSpaceGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # step size
    τ::AbstractFloat # tolerance for convergence
    generative_model::Union{Symbol, GenerativeModels.AbstractGenerativeModel} # generative model
    gm_params::NamedTuple # paramters for generative model
    gm_train::Bool # should generative model be trained at initialization
end

function LatentSpaceGenerator(
    ;
    loss::Symbol=:logitbinarycrossentropy,
    complexity::Function=norm,
    λ::AbstractFloat=0.1,
    ϵ::AbstractFloat=0.1,
    τ::AbstractFloat=1e-5,
    generative_model::Union{Symbol, GenerativeModels.AbstractGenerativeModel}=:VAE,
    gm_params::NamedTuple=(;),
    gm_train::Bool=true # by defaul generative model is trained at initialization
)
    if generative_model isa Symbol
        @assert generative_model ∈ names(GenerativeModels, all=true) "$generative_model does not specify a generative model."
    else
        @info "Using custom generative model." 
    end

    return LatentSpaceGenerator(loss, complexity, λ, ϵ, τ, generative_model, gm_params, gm_train)

end

# Loss:
using Flux
"""
    ℓ(generator::LatentSpaceGenerator, counterfactual_state::CounterfactualState)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(generator::LatentSpaceGenerator, counterfactual_state::CounterfactualState)

    output = :logits # currently counterfactual loss is always computed with respect to logits

    # 1. Encode counterfactual: z′ ∼ p(z|x)
    z′ = rand(VAE.encoder, counterfactual_state.x′)

    loss = getfield(Losses, generator.loss)(
        getfield(Models, output)(counterfactual_state.M, counterfactual_state.x′), 
        counterfactual_state.target_encoded
    )    

    return loss
end
