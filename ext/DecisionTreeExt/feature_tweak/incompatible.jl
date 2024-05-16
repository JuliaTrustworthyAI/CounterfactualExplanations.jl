"""
    Generators.incompatible(gen::FeatureTweakGenerator, ce::CounterfactualExplanation)

Overloads the `incompatible` function for the `FeatureTweakGenerator`.
"""
function Generators.incompatible(gen::FeatureTweakGenerator, ce::CounterfactualExplanation)
    incomp = false
    # Model compatibility
    if hasfield(typeof(ce.M), :type)
        mod = typeof(ce.M.type) <: CounterfactualExplanations.AbstractDecisionTree
    else
        # For legacy models that will be removed in the future
        mod = true
    end
    if !mod
        @warn "The `FeatureTweakGenerator` currently only supports tree models."
        incomp = true
    end
    # Feature encodings
    enc = !isnothing(ce.data.input_encoder)
    if enc
        @warn "The `FeatureTweakGenerator` is incompatible with feature encodings."
        incomp = true
    end
    return incomp
end