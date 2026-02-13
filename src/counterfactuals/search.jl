"""
    update!(ce::CounterfactualExplanation) 

An important subroutine that updates the counterfactual explanation. It takes a snapshot of the current counterfactual search state and passes it to the generator. Based on the current state the generator generates perturbations. Various constraints are then applied to the proposed vector of feature perturbations. Finally, the counterfactual search state is updated.
"""
function update!(ce::CounterfactualExplanation)

    # Generate peturbations:
    grad_ce_state = Generators.generate_perturbations(ce.generator, ce)
    grad_ce_state = apply_mutability(ce, grad_ce_state)         # mutability constraints
    counterfactual_state = ce.counterfactual_state + grad_ce_state                        # new proposed state

    # Updates:
    ce.counterfactual_state = counterfactual_state                                                  # update counterfactual
    ce.counterfactual = decode_state(ce)                                    # decoded counterfactual state
    apply_domain_constraints!(ce)                               # apply domain constraints
    _times_changed = reshape(
        decode_state(ce, grad_ce_state) .!= 0, size(ce.search[:times_changed_features])
    )
    ce.search[:times_changed_features] += _times_changed        # update number of times feature has been changed
    ce.search[:iteration_count] += 1                            # update iteration counter   
    ce.search[:path] = [ce.search[:path]..., ce.counterfactual_state]
    return terminated(ce)
end

"""
    apply_mutability(
        ce::CounterfactualExplanation,
        grad_ce_state::AbstractArray,
    )

A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
"""
function apply_mutability(ce::CounterfactualExplanation, grad_ce_state::AbstractArray)
    if typeof(ce.data.input_encoder) <: GenerativeModels.AbstractGenerativeModel ||
        typeof(ce.data.input_encoder) <: MultivariateStats.AbstractDimensionalityReduction
        if isnothing(ce.search)
            @warn "Mutability constraints not currently implemented for latent space search."
        end
        return grad_ce_state
    end

    mutability = ce.search[:mutability]

    # Create output array with same type as input (critical for type stability)
    result = similar(grad_ce_state)

    # Apply mutability constraints element-wise
    @inbounds for i in eachindex(mutability, grad_ce_state)
        case = mutability[i]
        val = grad_ce_state[i]

        # Use if-else instead of getfield for type stability
        result[i] = if case === :both
            val
        elseif case === :increase
            val < 0 ? zero(val) : val
        elseif case === :decrease
            val > 0 ? zero(val) : val
        elseif case === :none
            zero(val)
        else
            val  # fallback
        end
    end

    return result
end

"""
    apply_domain_constraints!(ce::CounterfactualExplanation)

Wrapper function that applies underlying domain constraints.
"""
function apply_domain_constraints!(ce::CounterfactualExplanation)
    ce.counterfactual = apply_domain_constraints(ce.data, ce.counterfactual)            # apply domain constraints in feature space
    return ce.counterfactual_state = encode_state(ce, ce.counterfactual)                             # re-encode counterfactual state
end
