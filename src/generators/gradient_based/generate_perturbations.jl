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

function propose_state(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    return propose_state(Models.Differentiability(ce.M), generator, ce)
end
"""
    propose_state(
        ::Models.IsDifferentiable,
        generator::AbstractGradientBasedGenerator,
        ce::AbstractCounterfactualExplanation,
    )

Proposes new state based on backpropagation for gradient-based generators and differentiable models.
"""
function propose_state(
    ::Models.IsDifferentiable,
    generator::AbstractGradientBasedGenerator,
    ce::AbstractCounterfactualExplanation,
)
    grads = ∇(generator, ce) # gradient
    new_s′ = deepcopy(ce.s′)
    Flux.Optimise.update!(generator.opt, new_s′, grads)
    return new_s′
end
