"""
A struct that collects all information relevant to a specific counterfactual explanation for a single individual.
"""
mutable struct TextCounterfactualExplanation <: AbstractCounterfactualExplanation
    x::AbstractArray
    target::RawTargetType
    target_encoded::EncodedTargetType
    # s′::AbstractArray
    x′::AbstractArray
    # data::DataPreprocessing.CounterfactualData
    M
    generator::Generators.AbstractGenerator
    # search::Union{Dict,Nothing}
    # convergence::AbstractConvergence
    num_counterfactuals::Int
    # initialization::Symbol
end

"""
	function CounterfactualExplanation(;
		x::AbstractArray,
		target::RawTargetType,
		data::CounterfactualData,
		M::Models.AbstractFittedModel,
		generator::Generators.AbstractGenerator,
		num_counterfactuals::Int = 1,
		initialization::Symbol = :add_perturbation,
        convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
	)

Outer method to construct a `CounterfactualExplanation` structure.
"""
function TextCounterfactualExplanation(
    x::AbstractArray,
    target::RawTargetType,
    # data::CounterfactualData,
    M,
    generator::Generators.AbstractGenerator;
    num_counterfactuals::Int=1,
    # initialization::Symbol=:add_perturbation,
    # convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
)
    # @assert any(predict_label(M, data) .== target) "Your model `M` never predicts the target value `target` for any of the samples contained in `data`. Are you sure the model is correctly specified?"
    # convergence = Convergence.get_convergence_type(convergence, data.y_levels)

    # Factual and target:
    # x = typeof(x) == Int ? select_factual(data, x) : x
    # target_encoded = data.output_encoder(target)

    # Instantiate:
    ce = TextCounterfactualExplanation(
        x,
        target,
        target_encoded,
        # x,
        x,
        # data,
        M,
        deepcopy(generator),
        # nothing,
        # convergence,
        num_counterfactuals,
        # initialization,
    )

    

    return ce
end
