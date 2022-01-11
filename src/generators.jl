# generators.jl
#
# Core package functionality that implements algorithmic recourse.

# --------------- Base type for generator:
using Flux

abstract type Generator end

# -------- Main method:
function generate_recourse(generator::Generator, x̅::AbstractArray, 𝓜::FittedModel, target::Float64; T=1000, 𝓘=[])
    
    # Setup and allocate memory:
    x̲ = copy(x̅) # start from factual
    D = length(x̲)
    path = reshape(x̲, 1, length(x̲)) # storing the path

    # Initialize:
    t = 1 # counter
    converged = convergence(generator, x̲, 𝓜, target, x̅) 

    # Search:
    while !converged && t < T 
        x̲ = step(generator, x̲, 𝓜, target, x̅, 𝓘)
        t += 1 # update number of times feature is changed
        converged = convergence(generator, x̲, 𝓜, target, x̅) # check if converged
        path = vcat(path, reshape(x̲, 1, D))
    end

    # Output:
    y̲ = round.(probs(𝓜, x̲))
    recourse = Recourse(x̲, y̲, path, generator, 𝓘, x̅, 𝓜, target) 
    
    return recourse
    
end

# --------------- Specific generators:

# -------- Wachter et al (2018): 
struct GenericGenerator <: Generator
    λ::Float64 # strength of penalty
    ϵ::Float64 # step size
    τ::Float64 # tolerance for convergence
end

ℓ(generator::GenericGenerator, x, 𝓜, t) = Flux.Losses.logitbinarycrossentropy(logits(𝓜, x), t)
complexity(generator::GenericGenerator, x̅, x̲) = norm(x̅-x̲)
objective(generator::GenericGenerator, x̲, 𝓜, t, x̅) = ℓ(generator, x̲, 𝓜, t) + generator.λ * complexity(generator, x̅, x̲) 
∇(generator::GenericGenerator, x̲, 𝓜, t, x̅) = gradient(() -> objective(generator, x̲, 𝓜, t, x̅), params(x̲))[x̲]

function step(generator::GenericGenerator, x̲, 𝓜, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    println(𝐠ₜ)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    println(𝐠ₜ)
    return x̲ - (generator.ϵ .* 𝐠ₜ)
end

function convergence(generator::GenericGenerator, x̲, 𝓜, t, x̅)
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    println(𝐠ₜ)
    all(abs.(𝐠ₜ) .< generator.τ)
end

# -------- Schut et al (2021):
struct GreedyGenerator <: Generator
    Γ::Float64 # desired level of confidence 
    δ::Float64 # perturbation size
    n::Int64 # maximum number of times any feature can be changed
end

ℓ(generator::GreedyGenerator, x, 𝓜, t) = - (t * log(𝛔(𝓜(x))) + (1-t) * log(1-𝛔(𝓜(x))))
objective(generator::GreedyGenerator, x̲, 𝓜, t) = ℓ(generator, x̲, 𝓜, t) 
∇(generator::GreedyGenerator, x̲, 𝓜, t) = gradient(() -> objective(generator, x̲, 𝓜, t), params(x̲))

function step(generator::GreedyGenerator, x̲, 𝓜, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    x̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return x̲
end

function convergence(generator::GreedyGenerator, x̲, 𝓜, t, x̅)
    𝓜.confidence(x̲) .> generator.Γ
end

