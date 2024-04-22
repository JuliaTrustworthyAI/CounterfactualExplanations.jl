include("feature_tweak.jl")

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(
    generator::FeatureTweakGenerator, ce::AbstractCounterfactualExplanation
)

    # Asserts related to https://github.com/JuliaTrustworthyAI/CounterfactualExplanations.jl/issues/258
    @assert isnothing(ce.data.input_encoder) "The `FeatureTweakGenerator` currently doesn't support feature encodings."
    @assert ce.generator.latent_space == false "The `FeatureTweakGenerator` currently doesn't support feature encodings."
    @assert isa(ce.M, Models.TreeModel) "The `FeatureTweakGenerator` currently only supports tree models. The counterfactual search will be terminated."

    s′ = deepcopy(ce.s′)
    new_s′ = feature_tweaking!(ce)
    Δs′ = new_s′ - s′                                           # gradient step
    Δs′ = _replace_nans(Δs′)
    Δs′ = convert.(eltype(ce.x), Δs′)

    return Δs′
end
