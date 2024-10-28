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
    Δcounterfactual_state = new_counterfactual_state - counterfactual_state
    Δcounterfactual_state = _replace_nans(Δcounterfactual_state)
    Δcounterfactual_state = convert.(eltype(ce.factual), Δcounterfactual_state)

    return Δcounterfactual_state
end
