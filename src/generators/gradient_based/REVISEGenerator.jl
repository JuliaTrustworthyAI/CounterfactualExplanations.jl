using LinearAlgebra

# -------- Joshi et al (2019): 
struct REVISEGenerator <: AbstractLatentSpaceGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # learning rate
    τ::AbstractFloat # tolerance for convergence
end

"""
    REVISEGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a REVISE generator.

# Examples
```julia-repl
generator = REVISEGenerator()
```
"""
REVISEGenerator(
    ;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Function=norm,
    λ::AbstractFloat=0.1,
    ϵ::AbstractFloat=0.1,
    τ::AbstractFloat=1e-5
) = REVISEGenerator(loss, complexity, λ, ϵ, τ)

# API streamlining:
using Parameters
@with_kw struct REVISEGeneratorParams
    ϵ::AbstractFloat=0.1
    τ::AbstractFloat=1e-5
end

REVISEGenerator(
    ;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Function=norm,
    λ::AbstractFloat=0.1,
    params::REVISEGeneratorParams
) = REVISEGenerator(loss, complexity, λ, params.ϵ, params.τ)