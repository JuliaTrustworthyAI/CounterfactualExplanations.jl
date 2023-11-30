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
    return Flux.gradient(() -> ℓ(generator, ce), Flux.params(ce.s′))[ce.s′]
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access.
"""
function ∂h(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    return Flux.gradient(() -> h(generator, ce), Flux.params(ce.s′))[ce.s′]
end

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators.
It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
If the counterfactual is being generated using Probe, the hinge loss is added to the gradient.
"""
function ∇(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractDifferentiableModel,
    ce::AbstractCounterfactualExplanation,
)
    return ∂ℓ(generator, M, ce) + ∂h(generator, ce) .+
           Convergence.hinge_loss(ce.convergence, ce)
end
