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
    FlattenedCE(factual::AbstractArray, target::RawTargetType)

If called on a `factual` and `target`, this outer constructor creates a `FlattenedCE` instance with the given `factual` and `target`, and `counterfactual` set to the `factual`.
"""
function FlattenedCE(factual::AbstractArray, target::RawTargetType)
    FlattenedCE(factual, target, factual)
end

"""
    (ce::CounterfactualExplanation)()::FlattenedCE

Calling the `ce::CounterfactualExplanation` object results in a [`FlattenedCE`](@ref) instance, which is the flattened version of the original.
"""
(ce::CounterfactualExplanation)()::FlattenedCE =
    FlattenedCE(ce.factual, ce.target, ce.counterfactual)

"""
    flatten(ce::CounterfactualExplanation)

Alias for `(ce::CounterfactualExplanation)()`. Converts a `CounterfactualExplanation` to its flattened form.
"""
flatten(ce::CounterfactualExplanation) = ce()

function unflatten(
    flat_ce::FlattenedCE,
    data::CounterfactualData,
    M::Models.AbstractModel,
    generator::Generators.AbstractGenerator;
    initialization::Symbol=:add_perturbation,
    convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
)::CounterfactualExplanation
    return CounterfactualExplanation(
        flat_ce.factual,
        flat_ce.target,
        data,
        M,
        generator;
        initialization=initialization,
        convergence=convergence,
        num_counterfactuals=num_counterfactuals(flat_ce),
    )
end

"""
    target_encoded(flat_ce::FlattenedCE, data::CounterfactualData)

Returns the encoded representation of `flat_ce.target`.
"""
function target_encoded(flat_ce::FlattenedCE, data::CounterfactualData)
    return data.output_encoder(flat_ce.target; y_levels=data.y_levels)
end

"""
    num_counterfactuals(flat_ce::FlattenedCE)

Returns the number of counterfactuals generated for `flat_ce` as implied by the data.
"""
num_counterfactuals(flat_ce::FlattenedCE) = size(flat_ce.counterfactual, 2)
