import DifferentiationInterface as DI

"A base type for a style of process."
abstract type PenaltyRequirements end

"By default, penalties have no extra requirements."
struct NoPenaltyRequirements <: PenaltyRequirements end
PenaltyRequirements(::Type) = NoPenaltyRequirements()

"Penalties that need access to neighbors in the target class."
struct NeedsNeighbours <: PenaltyRequirements end

"The `distance_from_target` method needs neighbors in the target class."
PenaltyRequirements(::Type{<:typeof(distance_from_target)}) = NeedsNeighbours()

# Implementing trait behaviour:
needs_neighbours(x::T) where {T} = needs_neighbours(PenaltyRequirements(T), x)
needs_neighbours(::NoPenaltyRequirements, x) = false
needs_neighbours(::NeedsNeighbours, x) = true

"""
    needs_neighbours(gen::AbstractGenerator)

Check if a generator needs access to neighbors in the target class.
"""
function needs_neighbours(gen::AbstractGenerator)
    penalty = CounterfactualExplanations.flatten_penalty(gen.penalty)
    return hasfield(typeof(gen), :penalty) ? any(needs_neighbours.(penalty)) : false
end

"""
    needs_neighbours(ce::AbstractCounterfactualExplanation)

Check if a counterfactual explanation needs access to neighbors in the target class.
"""
needs_neighbours(ce::AbstractCounterfactualExplanation) = needs_neighbours(ce.generator)

"A base type for AD backend requirements"
abstract type ADRequirements end

"By default, no special AD backend is required."
struct NoADRequirements <: ADRequirements end

ADRequirements(::Type) = NoADRequirements()

"This trait implies that `ForwardDiff` is required."
struct NeedsForwardDiff <: ADRequirements end

"The `hinge_loss` function requires `ForwardDiff`"
ADRequirements(::Type{<:typeof(hinge_loss)}) = NeedsForwardDiff()

choose_ad_backend(::NoADRequirements) = get_global_ad_backend()
choose_ad_backend(x::T) where {T} = choose_ad_backend(ADRequirements(T), x)
choose_ad_backend(::NoADRequirements, x) = get_global_ad_backend()
choose_ad_backend(::NeedsForwardDiff, x) = DI.AutoForwardDiff()

needs_special_backend(x::T) where {T} = needs_special_backend(ADRequirements(T), x)
needs_special_backend(::NoADRequirements, x) = false
needs_special_backend(::NeedsForwardDiff, x) = true

"""
    choose_ad_backend(gen::AbstractGenerator)

Choose an appropriate automatic differentiation backend for a given generator based on its penalty function. Handles both simple cases (no penalty function) and complex cases where multiple penalties might require different AD backends, ensuring that these backends are mutually exclusive.
"""
function choose_ad_backend(gen::AbstractGenerator)

    # Simplest case (no penalty function at all):
    if !hasfield(typeof(gen), :penalty)
        return choose_ad_backend(NoADRequirements())
    end

    # Flatten the penalty to ensure it's a single object or array of penalties
    penalty = CounterfactualExplanations.flatten_penalty(gen.penalty)

    # Check if any penalty requires a special AD backend
    if any(needs_special_backend.(penalty))
        @assert length(unique(choose_ad_backend.(penalty))) <= 2 "You have specified two or more penalties that require mutually exclusive AD backends."

        # Filter penalties that need special backends
        penalty = penalty[findall(needs_special_backend.(penalty))]

        # Choose the unique backend among those required by the filtered penalties
        return choose_ad_backend.(penalty) |> x -> unique(x)[1]
    else
        # If no special backend is needed, use the default AD backend
        return choose_ad_backend(NoADRequirements())
    end
end
