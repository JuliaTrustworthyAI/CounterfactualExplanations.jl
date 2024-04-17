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
    return Flux.gradient(ce -> ℓ(generator, ce), ce)[1][:s′]
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access. 

If the penalty is not provided, it returns 0.0. By default, Zygote never works out the gradient for constants and instead returns 'nothing', so we need to add a manual step to override this behaviour. See here: https://discourse.julialang.org/t/zygote-gradient/26715.
"""
function ∂h(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    _grad = Flux.gradient(ce -> h(generator, ce), ce)[1][:s′]
    _grad = isnothing(_grad) ? 0.0 : _grad
    return _grad
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
    return ∂ℓ(generator, M, ce) .+ ∂h(generator, ce) .+ hinge_loss(ce.convergence, ce)
end

"""
    hinge_loss(convergence::AbstractConvergence, ce::AbstractCounterfactualExplanation)

The default hinge loss for any convergence criterion.
Can be overridden inside the `Convergence` module as part of the definition of specific convergence criteria.
"""
function hinge_loss(convergence::AbstractConvergence, ce::AbstractCounterfactualExplanation)
    return 0
end
