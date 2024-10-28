"""
    update!(ce::CounterfactualExplanation) 

An important subroutine that updates the counterfactual explanation. It takes a snapshot of the current counterfactual search state and passes it to the generator. Based on the current state the generator generates perturbations. Various constraints are then applied to the proposed vector of feature perturbations. Finally, the counterfactual search state is updated.
"""
function update!(ce::CounterfactualExplanation)

    # Generate peturbations:
    Δcounterfactual_state = Generators.generate_perturbations(ce.generator, ce)
    Δcounterfactual_state = apply_mutability(ce, Δcounterfactual_state)         # mutability constraints
    counterfactual_state = ce.counterfactual_state + Δcounterfactual_state                        # new proposed state

    # Updates:
    ce.counterfactual_state = counterfactual_state                                                  # update counterfactual
    ce.counterfactual = decode_state(ce)                                    # decoded counterfactual state
    apply_domain_constraints!(ce)                               # apply domain constraints
    _times_changed = reshape(
        decode_state(ce, Δcounterfactual_state) .!= 0,
        size(ce.search[:times_changed_features]),
    )
    ce.search[:times_changed_features] += _times_changed        # update number of times feature has been changed
    ce.search[:iteration_count] += 1                            # update iteration counter   
    ce.search[:path] = [ce.search[:path]..., ce.counterfactual_state]
    return terminated(ce)
end

"""
    apply_mutability(
        ce::CounterfactualExplanation,
        Δcounterfactual_state::AbstractArray,
    )

A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
"""
function apply_mutability(
    ce::CounterfactualExplanation, Δcounterfactual_state::AbstractArray
)
    if typeof(ce.data.input_encoder) <: GenerativeModels.AbstractGenerativeModel ||
        typeof(ce.data.input_encoder) <: MultivariateStats.AbstractDimensionalityReduction
        if isnothing(ce.search)
            @warn "Mutability constraints not currently implemented for latent space search."
        end
        return Δcounterfactual_state
    end

    mutability = ce.search[:mutability]
    # Helper functions:
    both(x) = x
    increase(x) = ifelse(x < 0.0, 0.0, x)
    decrease(x) = ifelse(x > 0.0, 0.0, x)
    none(x) = 0.0
    cases = (both=both, increase=increase, decrease=decrease, none=none)

    # Apply:
    Δcounterfactual_state = map(
        (case, s) -> getfield(cases, case)(s), mutability, Δcounterfactual_state
    )

    return Δcounterfactual_state
end

"""
    apply_domain_constraints!(ce::CounterfactualExplanation)

Wrapper function that applies underlying domain constraints.
"""
function apply_domain_constraints!(ce::CounterfactualExplanation)
    ce.counterfactual = apply_domain_constraints(ce.data, ce.counterfactual)            # apply domain constraints in feature space
    return ce.counterfactual_state = encode_state(ce, ce.counterfactual)                             # re-encode counterfactual state
end
