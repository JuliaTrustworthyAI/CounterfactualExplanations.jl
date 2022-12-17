
################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State)
    gs = gradient(() -> ℓ(generator, counterfactual_state), Flux.params(counterfactual_state.s′))[counterfactual_state.s′]
    return gs
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State) = gradient(() -> h(generator, counterfactual_state), Flux.params(counterfactual_state.s′))[counterfactual_state.s′]

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators. It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
function ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State)
    ∂ℓ(generator, M, counterfactual_state) + ∂h(generator, counterfactual_state)
end

"""
    propose_state(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

Proposes new state based on backpropagation.
"""
function propose_state(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)
    grads = ∇(generator, counterfactual_state.M, counterfactual_state) # gradient
    new_s′ = deepcopy(counterfactual_state.s′)
    Flux.Optimise.update!(generator.opt, new_s′, grads)
    return new_s′
end

using Flux
"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State) 
    s′ = deepcopy(counterfactual_state.s′)
    new_s′ = propose_state(generator, counterfactual_state)
    Δs′ = new_s′ - s′ # gradient step
    return Δs′
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)
    mutability = counterfactual_state.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

"""
    conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
"""
function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state)
    status = all(abs.(𝐠ₜ) .< generator.τ) 
    return status
end

##################################################
# Specific Generators
##################################################

# Baseline
include("GenericGenerator.jl")          # Wachter et al. (2017)
include("GreedyGenerator.jl")           # Schut et al. (2021)
include("DICEGenerator.jl")             # Mothilal et al. (2020)
include("GravitationalGenerator.jl")    # Altmeyer et al. (2023)
include("ClapROARGenerator.jl")         # Altmeyer et al. (2023)

# Latent space
"""
    AbstractLatentSpaceGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators that search in a latent space. 
"""
abstract type AbstractLatentSpaceGenerator <: AbstractGradientBasedGenerator end

include("REVISEGenerator.jl") # Joshi et al. (2019)