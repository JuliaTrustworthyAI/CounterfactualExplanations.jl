"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    counterfactual_state = deepcopy(ce.counterfactual_state)
    new_counterfactual_state = propose_state(generator, ce)
    Δcounterfactual_state = new_counterfactual_state - counterfactual_state
    Δcounterfactual_state = _replace_nans(Δcounterfactual_state)
    Δcounterfactual_state = convert.(eltype(ce.factual), Δcounterfactual_state)
    Δcounterfactual_state *= ce.num_counterfactuals       # rescale to account for number of counterfactuals

    return Δcounterfactual_state
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
    new_counterfactual_state = deepcopy(ce.counterfactual_state)
    Flux.Optimise.update!(generator.opt, new_counterfactual_state, grads)
    return new_counterfactual_state
end
