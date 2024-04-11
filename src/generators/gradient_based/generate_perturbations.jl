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
    propose_state(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

Proposes new state based on backpropagation.
"""
function propose_state(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    println("loss:", ℓ(generator, ce))
    grads = ∇(generator, ce.M, ce) # gradient
    println("grads: ", grads)
    new_s′ = deepcopy(ce.s′)
    Flux.Optimise.update!(generator.opt, new_s′, grads)
    return new_s′
end
