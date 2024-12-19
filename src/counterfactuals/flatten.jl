"""
    FlattenedCE <: AbstractCounterfactualExplanation

A flattened representation of a `CounterfactualExplanation`, containing only the `factual`, `target`, and `counterfactual` attributes. This can be useful for compact storage or transmission of explanations.
"""
struct FlattenedCE <: AbstractCounterfactualExplanation
    factual::AbstractArray
    target::RawTargetType
    counterfactual::AbstractArray
end

"""
    (ce::CounterfactualExplanation)()::FlattenedCE

Calling the `ce::CounterfactualExplanation` object results in a [`FlattenedCE`](@ref) instance, which is the flattened version of the original.
"""
(ce::CounterfactualExplanation)()::FlattenedCE = FlattenedCE(ce.factual, ce.target, ce.counterfactual)

"""
    flatten(ce::CounterfactualExplanation)

Alias for `(ce::CounterfactualExplanation)()`. Converts a `CounterfactualExplanation` to its flattened form.
"""
flatten(ce::CounterfactualExplanation) = ce()