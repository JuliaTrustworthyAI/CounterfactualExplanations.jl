using LinearAlgebra

# -------- Mothilal et al. (2020): 
struct DiCEGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
using Parameters
@with_kw struct DiCEGeneratorParams
    ϵ::AbstractFloat=0.1
    τ::AbstractFloat=1e-5
end

"""
    DiCEGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = DiCEGenerator()
```
"""
DiCEGenerator(
    ;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Function=norm,
    λ::AbstractFloat=0.1,
    params::Union{NamedTuple,DiCEGeneratorParams}=DiCEGeneratorParams()
) = DiCEGenerator(loss, complexity, λ, params.ϵ, params.τ)

# Complexity:
# With thanks to various respondents here: https://discourse.julialang.org/t/getting-around-zygote-mutating-array-issue/83907/3
function ddp_diversity(counterfactual_state::State; perturbation_size=1e-5)
    X = counterfactual_state.s′
    xs = eachslice(X, dims = ndims(X))
    K = [1/(1 + norm(x .- y)) for x in xs, y in xs]
    K += Diagonal(randn(size(X,3))*perturbation_size)
    return det(K)
end

"""
    h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(generator::DiCEGenerator, counterfactual_state::CounterfactualState.State)
    dist_ = generator.complexity(counterfactual_state.x .- counterfactual_state.f(counterfactual_state.s′))
    ddp_ = ddp_diversity(counterfactual_state)
    return dist_ .- ddp_
end
