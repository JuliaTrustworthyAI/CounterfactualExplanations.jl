"""
    update!(ce::CounterfactualExplanation) 

An important subroutine that updates the counterfactual explanation. It takes a snapshot of the current counterfactual search state and passes it to the generator. Based on the current state the generator generates perturbations. Various constraints are then applied to the proposed vector of feature perturbations. Finally, the counterfactual search state is updated.
"""
function update!(ce::CounterfactualExplanation)

    # Generate peturbations:
    Δs′ = Generators.generate_perturbations(ce.generator, ce)
    Δs′ = apply_mutability(ce, Δs′)         # mutability constraints
    s′ = ce.s′ + Δs′                        # new proposed state

    # Updates:
    ce.s′ = s′                                                  # update counterfactual
    ce.x′ = decode_state(ce)                                    # decoded counterfactual state
    apply_domain_constraints!(ce)                               # apply domain constraints
    _times_changed = reshape(
        decode_state(ce, Δs′) .!= 0, size(ce.search[:times_changed_features])
    )
    ce.search[:times_changed_features] += _times_changed        # update number of times feature has been changed
    ce.search[:iteration_count] += 1                            # update iteration counter   
    ce.search[:path] = [ce.search[:path]..., ce.s′]
    ce.search[:loss] = [ce.search[:loss]..., Generators.total_loss(ce)]
    return terminated(ce)
end

"""
    apply_mutability(
        ce::CounterfactualExplanation,
        Δs′::AbstractArray,
    )

A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
"""
function apply_mutability(ce::CounterfactualExplanation, Δs′::AbstractArray)
    if typeof(ce.data.input_encoder) <: GenerativeModels.AbstractGenerativeModel ||
        typeof(ce.data.input_encoder) <: MultivariateStats.AbstractDimensionalityReduction
        if isnothing(ce.search)
            @warn "Mutability constraints not currently implemented for latent space search."
        end
        return Δs′
    end

    mutability = ce.search[:mutability]
    # Helper functions:
    both(x) = x
    increase(x) = ifelse(x < 0.0, 0.0, x)
    decrease(x) = ifelse(x > 0.0, 0.0, x)
    none(x) = 0.0
    cases = (both=both, increase=increase, decrease=decrease, none=none)

    # Apply:
    Δs′ = map((case, s) -> getfield(cases, case)(s), mutability, Δs′)

    return Δs′
end

"""
    apply_domain_constraints!(ce::CounterfactualExplanation)

Wrapper function that applies underlying domain constraints.
"""
function apply_domain_constraints!(ce::CounterfactualExplanation)
    ce.x′ = apply_domain_constraints(ce.data, ce.x′)            # apply domain constraints in feature space
    return ce.s′ = encode_state(ce, ce.x′)                             # re-encode counterfactual state
end
