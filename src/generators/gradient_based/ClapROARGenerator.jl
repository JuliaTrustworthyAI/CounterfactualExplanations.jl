using CounterfactualExplanations
using CounterfactualExplanations.Generators
using Flux
using LinearAlgebra
using Parameters
using Statistics

mutable struct ClaPROARGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::Union{AbstractFloat,AbstractVector} # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat}
    opt::Flux.Optimise.AbstractOptimiser # optimizer
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
@with_kw struct ClaPROARGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
end

"""
    ClaPROARGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        opt::Flux.Optimise.AbstractOptimiser=Flux.Optimise.Descent(),
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = ClaPROARGenerator()
```
"""
function ClaPROARGenerator(;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Function=norm,
    λ::Union{AbstractFloat,AbstractVector}=[0.1, 1.0],
    decision_threshold=nothing,
    kwargs...
)
    params = ClaPROARGeneratorParams(; kwargs...)
    ClaPROARGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ)
end

"""
    gradient_penalty(
        generator::ClaPROARGenerator,
        counterfactual_explanation::AbstractCounterfactualExplanation,
    )

Additional penalty for ClaPROARGenerator.
"""
function gradient_penalty(
    generator::ClaPROARGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)

    x_ = CounterfactualExplanations.decode_state(counterfactual_explanation)
    M = counterfactual_explanation.M
    model = isa(M.model, Vector) ? M.model : [M.model]
    y_ = counterfactual_explanation.target_encoded

    if M.likelihood == :classification_binary
        loss_type = :logitbinarycrossentropy
    else
        loss_type = :logitcrossentropy
    end

    loss(x, y) =
        sum([getfield(Flux.Losses, loss_type)(nn(x), y) for nn in model]) / length(model)

    return loss(x_, y_)
end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function Generators.h(
    generator::ClaPROARGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)

    # Distance from factual:
    dist_ = generator.complexity(
        counterfactual_explanation.x .-
        CounterfactualExplanations.decode_state(counterfactual_explanation),
    )

    # Euclidean norm of gradient:
    in_target_domain = all(target_probs(counterfactual_explanation) .>= 0.5)
    if in_target_domain
        grad_norm = gradient_penalty(generator, counterfactual_explanation)
    else
        grad_norm = 0
    end

    if length(generator.λ) == 1
        penalty = generator.λ * (dist_ .+ grad_norm)
    else
        penalty = generator.λ[1] * dist_ .+ generator.λ[2] * grad_norm
    end
    return penalty
end
