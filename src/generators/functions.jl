# --------------- Base type for generator:
"""
    Generator

An abstract type that serves as the base type for recourse generators. 
"""
abstract type Generator end

# --------------- Specific generators:

# -------- Wachter et al (2018): 
"""
    GenericGenerator(λ::Float64, ϵ::Float64, τ::Float64, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})

A constructor for a generic recourse generator. It takes values for the complexity penalty `λ`, the learning rate `ϵ`, the tolerance for convergence `τ`, 
    the type of `loss` function to be used in the recourse objective and a mutability constraint mask `𝑭`.

# Examples
```julia-repl
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
```

See also [`generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000)`](@ref).
"""
struct GenericGenerator <: Generator
    λ::Float64 # strength of penalty
    ϵ::Float64 # step size
    τ::Float64 # tolerance for convergence
    loss::Symbol # loss function
    𝑭::Union{Nothing,Vector{Symbol}} # mutibility constraints 
end

ℓ(generator::GenericGenerator, x̲, 𝓜, t) = getfield(Losses, generator.loss)(Models.logits(𝓜, x̲), t)
complexity(x̅, x̲) = norm(x̅-x̲)
objective(generator::GenericGenerator, x̲, 𝓜, t, x̅) = ℓ(generator, x̲, 𝓜, t) + generator.λ * complexity(x̅, x̲) 
∇(generator::GenericGenerator, x̲, 𝓜, t, x̅) = gradient(() -> objective(generator, x̲, 𝓜, t, x̅), params(x̲))[x̲]

function generate_perturbations(generator::GenericGenerator, x̲, 𝓜, t, x̅) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅) # gradient
    Δx̲ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx̲
end

function mutability_constraints(generator::GenericGenerator, 𝑷)
    d = length(𝑷)
    if isnothing(generator.𝑭)
        𝑭 = [:both for i in 1:d]
    else 
        𝑭 = generator.𝑭
    end
    return 𝑭
end 

function condtions_satisified(generator::GenericGenerator, x̲, 𝓜, t, x̅)
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    all(abs.(𝐠ₜ) .< generator.τ) 
end

# -------- Schut et al (2021):
"""
    GreedyGenerator(δ::Float64, n::Int64, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})

Constructs a greedy recourse generator for Bayesian models. It takes values for the perturbation size `δ`, the maximum number of times `n` that any feature can be changed, 
    the type of `loss` function to be used in the recourse objective and a mutability constraint mask `𝑭`.

# Examples
```julia-repl
generator = GreedyGenerator(0.01,20,:logitbinarycrossentropy, nothing)
```

See also [`generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::Models.FittedModel, target::Float64; T=1000)`](@ref).
"""
struct GreedyGenerator <: Generator
    δ::Float64 # perturbation size
    n::Int64 # maximum number of times any feature can be changed
    loss::Symbol # loss function
    𝑭::Union{Nothing,Vector{Symbol}} # mutibility constraints 
end

objective(generator::GreedyGenerator, x̲, 𝓜, t) = getfield(Losses, generator.loss)(Models.logits(𝓜, x̲), t)
∇(generator::GreedyGenerator, x̲, 𝓜, t) = gradient(() -> objective(generator, x̲, 𝓜, t), params(x̲))[x̲]

function generate_perturbations(generator::GreedyGenerator, x̲, 𝓜, t, x̅) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t) # gradient
    Δx̲ = reshape(zeros(length(x̲)), size(𝐠ₜ))
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    Δx̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return Δx̲
end

function mutability_constraints(generator::GreedyGenerator, 𝑷)
    d = length(𝑷)
    if isnothing(generator.𝑭)
        𝑭 = [:both for i in 1:d]
    else 
        𝑭 = generator.𝑭
    end
    𝑭[𝑷 .>= generator.n] .= :none
    return 𝑭
end 

function condtions_satisified(generator::GreedyGenerator, x̲, 𝓜, t, x̅)
    return true # Greedy generator only requires confidence threshold to be met
end