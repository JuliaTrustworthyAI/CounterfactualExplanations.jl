"""
    Generators.incompatible(
        gen::FeatureTweakGenerator,
        M::Models.AbstractModel,
        data::CounterfactualData,
    )

Overloads the `incompatible` function for the `FeatureTweakGenerator`.
"""
function Generators.incompatible(
    gen::FeatureTweakGenerator,
    M::Models.AbstractModel,
    data::CounterfactualData,
)
    incomp = false
    # Model compatibility
    if hasfield(typeof(M), :type)
        mod = typeof(M.type) <: CounterfactualExplanations.AbstractDecisionTree
    else
        # For legacy models that will be removed in the future
        mod = true
    end
    if !mod
        @warn "The `FeatureTweakGenerator` currently only supports tree models."
        incomp = true
    end
    # Feature encodings
    enc = !isnothing(data.input_encoder)
    if enc
        @warn "The `FeatureTweakGenerator` is incompatible with feature encodings."
        incomp = true
    end
    return incomp
end
