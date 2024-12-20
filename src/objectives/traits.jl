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
needs_neighbours(gen::AbstractGenerator) =
    hasfield(typeof(gen), :penalty) ? any(needs_neighbours.(gen.penalty)) : false

"""
    needs_neighbours(ce::AbstractCounterfactualExplanation)

Check if a counterfactual explanation needs access to neighbors in the target class.
"""
needs_neighbours(ce::AbstractCounterfactualExplanation) = needs_neighbours(ce.generator)
