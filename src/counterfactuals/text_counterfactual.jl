"""
A placeholder struct that collects all information relevant to a specific text counterfactual explanation.
"""
mutable struct TextCounterfactualExplanation <: AbstractCounterfactualExplanation
    x::AbstractArray
    target::RawTargetType
    target_encoded::EncodedTargetType
    x′::AbstractArray
    M
    generator::Generators.AbstractGenerator
    num_counterfactuals::Int
end

"""
	function TextCounterfactualExplanation(;
		x::AbstractArray,
		target::RawTargetType,
		M::Models.AbstractFittedModel,
		generator::Generators.AbstractGenerator,
		num_counterfactuals::Int = 1,
	)

Outer method to construct a `TextCounterfactualExplanation` structure.
"""
function TextCounterfactualExplanation(
    x::AbstractArray,
    target::RawTargetType,
    M,
    generator::Generators.AbstractGenerator;
    num_counterfactuals::Int=1,
)
  
    ce = TextCounterfactualExplanation(
        x,
        target,
        target_encoded,
        x,
        M,
        deepcopy(generator),
        num_counterfactuals,
    )

    

    return ce
end
