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
    params::Dict
    search::Union{Dict,Nothing}
    convergence::AbstractConvergenceType
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
		max_iter::Int = 100,
		num_counterfactuals::Int = 1,
		initialization::Symbol = :add_perturbation,
        converge_when::Symbol=:decision_threshold,
        convergence::AbstractConvergenceType=nothing,
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
    converge_when::Symbol=:decision_threshold,
    convergence::Union{AbstractConvergenceType,Nothing}=nothing,
)

    # Assertions:
    @assert any(predict_label(M, data) .== target) "You model `M` never predicts the target value `target` for any of the samples contained in `data`. Are you sure the model is correctly specified?"
    @assert converge_when ∈ [
        :decision_threshold,
        :generator_conditions,
        :max_iter,
        :invalidation_rate,
        :early_stopping,
    ] "Convergence criterion not recognized: $converge_when."

    if isnothing(convergence)
        convergence = convergence_catalogue[converge_when]
    end

    # Factual:
    x = typeof(x) == Int ? select_factual(data, x) : x

    # Target:
    target_encoded = data.output_encoder(target)

    # Initial Parameters:
    params = Dict{Symbol,Any}(
        :mutability => DataPreprocessing.mutability_constraints(data),
        :latent_space => generator.latent_space,
        :dim_reduction => generator.dim_reduction,
    )
    ids = findall(predict_label(M, data) .== target)
    n_candidates = minimum([size(data.y, 2), 1000])
    candidates = select_factual(data, rand(ids, n_candidates))
    params[:potential_neighbours] = reduce(hcat, map(x -> x[1], collect(candidates)))

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
        params,
        nothing,
        convergence,
        num_counterfactuals,
        initialization,
    )

    # Initialization:
    adjust_shape!(ce)                   # adjust shape to specified number of counterfactuals
    ce.s′ = encode_state(ce)            # encode the counterfactual state
    ce.s′ = initialize_state(ce)        # initialize the counterfactual state
    ce.x′ = decode_state(ce)            # decode the counterfactual state

    # Initialize search:
    ce.search = Dict(
        :iteration_count => 0,
        :times_changed_features => zeros(size(decode_state(ce))),
        :path => [ce.s′],
    )

    # This is lifted out of the above ce.search initialization because calling converged(ce) might self-reference
    # the above fields, which are not yet initialized.
    ce.search[:converged] = converged(ce)
    ce.search[:terminated] = terminated(ce)

    # Check for redundancy:
    if in_target_class(ce) && threshold_reached(ce)
        @info "Factual already in target class and probability exceeds threshold γ."
    end

    return ce
end
