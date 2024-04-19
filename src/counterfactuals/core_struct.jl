using ..GenerativeModels: GenerativeModels
using MultivariateStats: MultivariateStats

"""
A struct that collects all information relevant to a specific counterfactual explanation for a single individual.
"""
mutable struct CounterfactualExplanation <: AbstractCounterfactualExplanation
    x::AbstractArray
    target::RawTargetType
    target_encoded::EncodedTargetType
    s′::AbstractArray
    x′::AbstractArray
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractFittedModel
    generator::Generators.AbstractGenerator
    search::Union{Dict,Nothing}
    convergence::AbstractConvergence
    num_counterfactuals::Int
    initialization::Symbol
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
function CounterfactualExplanation(
    x::AbstractArray,
    target::RawTargetType,
    data::CounterfactualData,
    M::Models.AbstractFittedModel,
    generator::Generators.AbstractGenerator;
    num_counterfactuals::Int=1,
    initialization::Symbol=:add_perturbation,
    convergence::Union{AbstractConvergence,Symbol}=:decision_threshold,
)

    # Assertions:
    @assert any(predict_label(M, data) .== target) "Your model `M` never predicts the target value `target` for any of the samples contained in `data`. Are you sure the model is correctly specified?"

    # Setups:
    convergence = Convergence.get_convergence_type(convergence, data.y_levels)
    if generator.latent_space && !(typeof(data.input_encoder) <: GenerativeModels.AbstractGenerativeModel)
        @info "No pre-trained generative model found. Training default VAE."
        data.input_encoder = DataPreprocessing.fit_transformer(data, GenerativeModels.VAE)
    end
    if generator.dim_reduction && !(typeof(data.input_encoder) <: MultivariateStats.AbstractDimensionalityReduction)
        @info "No pre-trained dimensionality reduction model found. Training default PCA."
        data.input_encoder = DataPreprocessing.fit_transformer(data, MultivariateStats.PCA)
    end

    # Factual and target:
    x = typeof(x) == Int ? select_factual(data, x) : x
    target_encoded = data.output_encoder(target)

    # Instantiate:
    ce = CounterfactualExplanation(
        x,
        target,
        target_encoded,
        x,
        x,
        data,
        M,
        deepcopy(generator),
        nothing,
        convergence,
        num_counterfactuals,
        initialization,
    )

    # Initialize search:
    ce.search = Dict(
        :iteration_count => 0,
        :mutability => DataPreprocessing.mutability_constraints(data),
        :potential_neighbours => find_potential_neighbours(ce),
    )

    # Initialization:
    adjust_shape!(ce) |> encode_state! |> initialize_state! |> decode_state!

    ce.search[:path] = [ce.s′]
    ce.search[:times_changed_features] = zeros(size(decode_state(ce)))

    # Check for redundancy:
    if in_target_class(ce) && Convergence.threshold_reached(ce)
        @info "Factual already in target class and probability exceeds threshold γ."
    end

    return ce
end
