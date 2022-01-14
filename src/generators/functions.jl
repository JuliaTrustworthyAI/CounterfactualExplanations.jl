# --------------- Base type for generator:
"""
    Generator

An abstract type that serves as the base type for recourse generators. 
"""
abstract type Generator end

# --------------- Specific generators:

# -------- Wachter et al (2018): 
"""
    GenericGenerator(λ::Float64, ϵ::Float64, τ::Float64, loss::Symbol)

A constructor for a generic recourse generator. 
It takes values for the complexity penalty `λ`, the learning rate `ϵ`, the tolerance for convergence `τ` and the type of `loss` function to be used in the recourse objective. 

# Examples
```julia-repl
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy)
```

See also [`generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000, 𝓘=[])`](@ref).
"""
struct GenericGenerator <: Generator
    λ::Float64 # strength of penalty
    ϵ::Float64 # step size
    τ::Float64 # tolerance for convergence
    loss::Symbol # loss function
end

ℓ(generator::GenericGenerator, x̲, 𝓜, t) = getfield(Losses, generator.loss)(Models.logits(𝓜, x̲), t)
complexity(x̅, x̲) = norm(x̅-x̲)
objective(generator::GenericGenerator, x̲, 𝓜, t, x̅) = ℓ(generator, x̲, 𝓜, t) + generator.λ * complexity(x̅, x̲) 
∇(generator::GenericGenerator, x̲, 𝓜, t, x̅) = gradient(() -> objective(generator, x̲, 𝓜, t, x̅), params(x̲))[x̲]

function update_recourse(generator::GenericGenerator, x̲, 𝓜, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    return x̲ - (generator.ϵ .* 𝐠ₜ)
end

function convergence(generator::GenericGenerator, x̲, 𝓜, γ, t, x̅)
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    all(abs.(𝐠ₜ) .< generator.τ) || abs(Models.probs(𝓜, x̲)[1] - t) <= abs(t-γ)
end

# -------- Schut et al (2021):
"""
    GreedyGenerator(δ::Float64, n::Int64, loss::Symbol)

Constructs a greedy recourse generator for Bayesian models.
It takes values for the perturbation size `δ`, the maximum number of times `n` that any feature can be changed 
and the type of `loss` function to be used in the recourse objective. 

# Examples
```julia-repl
generator = GreedyGenerator(0.01,20,:logitbinarycrossentropy)
```

See also [`generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000, 𝓘=[])`](@ref).
"""
struct GreedyGenerator <: Generator
    δ::Float64 # perturbation size
    n::Int64 # maximum number of times any feature can be changed
    loss::Symbol # loss function
end

objective(generator::GreedyGenerator, x̲, 𝓜, t) = getfield(Losses, generator.loss)(Models.logits(𝓜, x̲), t)
∇(generator::GreedyGenerator, x̲, 𝓜, t) = gradient(() -> objective(generator, x̲, 𝓜, t), params(x̲))[x̲]

function update_recourse(generator::GreedyGenerator, x̲, 𝓜, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    x̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return x̲
end

function convergence(generator::GreedyGenerator, x̲, 𝓜, γ, t, x̅)
    abs(Models.probs(𝓜, x̲)[1] - t) <= abs(t-γ)
end