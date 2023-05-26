using Flux
using Statistics

"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractDifferentiableModel,
    ce::AbstractCounterfactualExplanation,
)
    gs = 0
    if (ce.convergence[:converge_when] == :invalidation_rate)
        gs =
            gradient(() -> ℓ(generator, ce), Flux.params(ce.s′))[ce.s′] .+
            hinge_loss(ce)
    else
        gs = gradient(() -> ℓ(generator, ce), Flux.params(ce.s′))[ce.s′]
    end
    return gs
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access.
"""
function ∂h(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    return gradient(() -> h(generator, ce), Flux.params(ce.s′))[ce.s′]
end

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators.
It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
function ∇(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractDifferentiableModel,
    ce::AbstractCounterfactualExplanation,
)
    return ∂ℓ(generator, M, ce) + ∂h(generator, ce)
end

"""
    propose_state(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

Proposes new state based on backpropagation.
"""
function propose_state(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    grads = ∇(generator, ce.M, ce) # gradient
    new_s′ = deepcopy(ce.s′)    
    Flux.Optimise.update!(generator.opt, new_s′, grads)
    return new_s′
end

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    s′ = deepcopy(ce.s′)
    new_s′ = propose_state(generator, ce)
    Δs′ = new_s′ - s′                                           # gradient step
    Δs′ = _replace_nans(Δs′)
    Δs′ *= ce.num_counterfactuals       # rescale to account for number of counterfactuals
    Δs′ = convert.(eltype(ce.x), Δs′)

    return Δs′
end

"""
    _replace_nans(Δs′::AbstractArray, old_new::Pair=(NaN => 0))

Helper function to deal with exploding gradients. This is only a temporary fix and will be improved.
"""
function _replace_nans(Δs′::AbstractArray, old_new::Pair=(NaN => 0))
    return replace(Δs′, old_new)
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    mutability = ce.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end

"""
    conditions_satisfied(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators. By default, gradient-based search is considered to have converged as soon as the proposed feature changes for all features are smaller than one percent of its standard deviation.
"""
function conditions_satisfied(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    Δs′ = ∇(generator, ce.M, ce)
    Δs′ = CounterfactualExplanations.apply_mutability(ce, Δs′)
    τ = ce.convergence[:gradient_tol]
    satisfied = map(x -> all(abs.(x) .< τ), eachslice(Δs′; dims=ndims(Δs′)))
    success_rate = sum(satisfied) / ce.num_counterfactuals
    status = success_rate > ce.convergence[:min_success_rate]
    return status
end
