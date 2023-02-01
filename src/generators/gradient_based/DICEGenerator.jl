using LinearAlgebra
using Parameters

# -------- Following Mothilal et al. (2020): 
mutable struct DiCEGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::Union{AbstractFloat,AbstractVector} # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat} # probability threshold
    opt::Flux.Optimise.AbstractOptimiser # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
@with_kw struct DiCEGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
end

"""
    DiCEGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=LinearAlgebra.norm,
        λ::AbstractFloat=0.1,
        opt::Flux.Optimise.AbstractOptimiser=Flux.Optimise.Descent(),
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = DiCEGenerator()
```
"""
function DiCEGenerator(;
    loss::Union{Nothing,Symbol} = nothing,
    complexity::Function = norm,
    λ::Union{AbstractFloat,AbstractVector} = [0.1, 1.0],
    decision_threshold = nothing,
    kwargs...,
)
    params = DiCEGeneratorParams(; kwargs...)
    DiCEGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ)
end

# Complexity:
# With thanks to various respondents here: https://discourse.julialang.org/t/getting-around-zygote-mutating-array-issue/83907/3
function ddp_diversity(
    counterfactual_explanation::AbstractCounterfactualExplanation;
    perturbation_size = 1e-5,
)
    X = counterfactual_explanation.s′
    xs = eachslice(X, dims = ndims(X))
    K = [1 / (1 + LinearAlgebra.norm(x .- y)) for x in xs, y in xs]
    K += Diagonal(randn(size(X, 3)) * perturbation_size)
    return det(K)
end

"""
    h(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(
    generator::DiCEGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)
    dist_ = generator.complexity(
        counterfactual_explanation.x .-
        CounterfactualExplanations.decode_state(counterfactual_explanation),
    )
    ddp_ = ddp_diversity(counterfactual_explanation)
    if length(generator.λ) == 1
        penalty = generator.λ * (dist_ .- ddp_)
    else
        penalty = generator.λ[1] * dist_ .- generator.λ[2] * ddp_
    end
    return penalty
end
