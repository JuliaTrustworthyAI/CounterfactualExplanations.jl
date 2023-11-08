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