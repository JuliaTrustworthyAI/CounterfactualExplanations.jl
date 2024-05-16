"""
    generate_perturbations(
        generator::AbstractGenerator, ce::AbstractCounterfactualExplanation
    )

The default method to generate feature perturbations for any generator.
"""
function generate_perturbations(
    generator::AbstractGenerator, ce::AbstractCounterfactualExplanation
)
    s′ = deepcopy(ce.s′)
    new_s′ = propose_state(generator, ce)
    Δs′ = new_s′ - s′
    Δs′ = _replace_nans(Δs′)
    Δs′ = convert.(eltype(ce.x), Δs′)

    return Δs′
end
