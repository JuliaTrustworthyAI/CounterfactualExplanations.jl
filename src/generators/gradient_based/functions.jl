
################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_explanation::AbstractCounterfactualExplanation)
    gs = gradient(() -> ℓ(generator, counterfactual_explanation), Flux.params(counterfactual_explanation.s′))[counterfactual_explanation.s′]
    return gs
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂h(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation) = gradient(() -> h(generator, counterfactual_explanation), Flux.params(counterfactual_explanation.s′))[counterfactual_explanation.s′]

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators. It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
function ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_explanation::AbstractCounterfactualExplanation)
    ∂ℓ(generator, M, counterfactual_explanation) + ∂h(generator, counterfactual_explanation)
end

"""
    propose_state(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

Proposes new state based on backpropagation.
"""
function propose_state(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)
    grads = ∇(generator, counterfactual_explanation.M, counterfactual_explanation) # gradient
    new_s′ = deepcopy(counterfactual_explanation.s′)
    Flux.Optimise.update!(generator.opt, new_s′, grads)
    return new_s′
end

using Flux
"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation) 
    s′ = deepcopy(counterfactual_explanation.s′)
    new_s′ = propose_state(generator, counterfactual_explanation)
    Δs′ = new_s′ - s′ # gradient step
    return Δs′
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)
    mutability = counterfactual_explanation.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

"""
    conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
"""
function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)
    𝐠ₜ = ∇(generator, counterfactual_explanation.M, counterfactual_explanation)
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