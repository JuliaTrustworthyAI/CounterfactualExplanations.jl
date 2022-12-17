using LinearAlgebra
using Parameters

# -------- Joshi et al (2019): 
mutable struct REVISEGenerator <: AbstractLatentSpaceGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat} # probability threshold
    opt::Any # learning rate
    τ::AbstractFloat # tolerance for convergence
end

# API streamlining:
@with_kw struct REVISEGeneratorParams
    opt::Any = Flux.Optimise.Descent()
    τ::AbstractFloat = 1e-5
end

"""
    REVISEGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=LinearAlgebra.norm,
        λ::AbstractFloat=0.1,
        opt::Any=Flux.Optimise.Descent(),
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a REVISE generator.

# Examples
```julia-repl
generator = REVISEGenerator()
```
"""
function REVISEGenerator(;
    loss::Union{Nothing,Symbol} = nothing,
    complexity::Function = LinearAlgebra.norm,
    λ::AbstractFloat = 0.1,
    decision_threshold = 0.5,
    kwargs...,
)
    params = REVISEGeneratorParams(; kwargs...)
    REVISEGenerator(loss, complexity, λ, decision_threshold, params.opt, params.τ)
end
