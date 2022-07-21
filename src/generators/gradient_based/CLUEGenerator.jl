using LinearAlgebra

# -------- Antoran et al (2020): 
struct CLUEGenerator <: AbstractLatentSpaceGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
using Parameters
@with_kw struct CLUEGeneratorParams
    ϵ::AbstractFloat=0.1
    τ::AbstractFloat=1e-5
end

"""
    CLUEGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a CLUE generator.

# Examples
```julia-repl
generator = CLUEGenerator()
```
"""
function CLUEGenerator(;loss::Union{Nothing,Symbol}=nothing,complexity::Function=norm,λ::AbstractFloat=0.1,kwargs...)
    @info "CLUE is meant to be used with Bayesian classifiers."
    params = CLUEGeneratorParams(;kwargs...)
    CLUEGenerator(loss, complexity, λ, params.ϵ, params.τ)
end

# using Flux
# """
#     ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

# The default method to apply the generator loss function to the current counterfactual state for any generator.
# """
# function ℓ(generator::CLUEGenerator, counterfactual_state::CounterfactualState.State) 
#     loss = predictive_entropy(counterfactual_state.M, counterfactual_state.f(counterfactual_state.s′))
#     return loss
# end

# using Statistics
# function predictive_entropy(model::Models.AbstractFittedModel, X::AbstractArray; agg=mean)
#     p = probs(model,X)
#     output = agg(sum(@.(p * log(p)),dims=2))
#     return output
# end