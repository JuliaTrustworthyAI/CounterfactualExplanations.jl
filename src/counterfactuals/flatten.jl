"""
    FlattenedCE <: AbstractCounterfactualExplanation

A flattened representation of a `CounterfactualExplanation`, containing only the `factual`, `target`, and `counterfactual` attributes. This can be useful for compact storage or transmission of explanations.
"""
struct FlattenedCE <: AbstractCounterfactualExplanation
    factual::AbstractArray
    target::RawTargetType
    counterfactual_state::AbstractArray
    counterfactual::AbstractArray
    search::Dict
end

"""
    (ce::CounterfactualExplanation)()::FlattenedCE

Calling the `ce::CounterfactualExplanation` object results in a [`FlattenedCE`](@ref) instance, which is the flattened version of the original.
"""
function (ce::CounterfactualExplanation)(; store_path::Bool=false)::FlattenedCE
    search_dict = ce.search
    if !store_path
        search_dict[:path] = nothing
    end
    return FlattenedCE(
        ce.factual, ce.target, ce.counterfactual_state, ce.counterfactual, search_dict
    )
end

"""
    flatten(ce::CounterfactualExplanation)

Alias for `(ce::CounterfactualExplanation)()`. Converts a `CounterfactualExplanation` to its flattened form.
"""
flatten(ce::CounterfactualExplanation; kwrgs...) = ce(; kwrgs...)

function unflatten(
    flat_ce::FlattenedCE,
    data::CounterfactualData,
    M::Models.AbstractModel,
    generator::Generators.AbstractGenerator;
    initialization::Symbol=:add_perturbation,
    convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
)::CounterfactualExplanation
    ce = CounterfactualExplanation(
        flat_ce.factual,
        flat_ce.target,
        target_encoded(flat_ce, data),
        flat_ce.counterfactual_state,
        flat_ce.counterfactual,
        data,
        M,
        generator,
        flat_ce.search,
        get_convergence_type(convergence, data.y_levels),
        size(flat_ce.counterfactual, 2),
        initialization,
    )
    adjust_shape!(ce)
    return ce
end

"""
    target_encoded(flat_ce::FlattenedCE, data::CounterfactualData)

Returns the encoded representation of `flat_ce.target`.
"""
function target_encoded(flat_ce::FlattenedCE, data::CounterfactualData)
    return data.output_encoder(flat_ce.target; y_levels=data.y_levels)
end
