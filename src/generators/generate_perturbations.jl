"""
    generate_perturbations(
        generator::AbstractGenerator, ce::AbstractCounterfactualExplanation
    )

The default method to generate feature perturbations for any generator.
"""
function generate_perturbations(
    generator::AbstractGenerator, ce::AbstractCounterfactualExplanation
)
    counterfactual_state = deepcopy(ce.counterfactual_state)
    new_counterfactual_state = propose_state(generator, ce)
    grad_ce_state = new_counterfactual_state - counterfactual_state
    grad_ce_state = _replace_nans(grad_ce_state)
    grad_ce_state = convert.(eltype(ce.factual), grad_ce_state)

    return grad_ce_state
end
