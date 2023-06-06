# using ChainRulesCore
# using Flux
# using MLUtils
# using SliceMap
# using Statistics
# using StatsBase

# """
# A struct that collects all information relevant to a specific counterfactual explanation for a single individual.
# """
# mutable struct CounterfactualExplanation <: AbstractCounterfactualExplanation
#     x::AbstractArray
#     target::RawTargetType
#     target_encoded::EncodedTargetType
#     s′::AbstractArray
#     data::DataPreprocessing.CounterfactualData
#     M::Models.AbstractFittedModel
#     generator::Generators.AbstractGenerator
#     generative_model_params::NamedTuple
#     params::Dict
#     search::Union{Dict,Nothing}
#     convergence::Dict
#     num_counterfactuals::Int
#     initialization::Symbol
# end

<<<<<<< HEAD
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
		generative_model_params::NamedTuple = (;),
		min_success_rate::AbstractFloat=0.99,
        converge_when::Symbol=:decision_threshold,
        invalidation_rate::AbstractFloat=0.5,
        learning_rate::AbstractFloat=1.0,
        variance::AbstractFloat=0.01,
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
    generative_model_params::NamedTuple=(;),
    max_iter::Int=100,
    decision_threshold::AbstractFloat=0.5,
    gradient_tol::AbstractFloat=parameters[:τ],
    min_success_rate::AbstractFloat=parameters[:min_success_rate],
    converge_when::Symbol=:decision_threshold,
    invalidation_rate::AbstractFloat=0.5,
    learning_rate::AbstractFloat=1.0,
    variance::AbstractFloat=0.01,
)

    # Assertions:
    @assert any(predict_label(M, data) .== target) "You model `M` never predicts the target value `target` for any of the samples contained in `data`. Are you sure the model is correctly specified?"
    @assert 0.0 < min_success_rate <= 1.0 "Minimum success rate should be ∈ [0.0,1.0]."
    @assert converge_when ∈
        [:decision_threshold, :generator_conditions, :max_iter, :invalidation_rate]
=======
# """
#     function CounterfactualExplanation(;
#         x::AbstractArray,
#         target::RawTargetType,
#         data::CounterfactualData,
#         M::Models.AbstractFittedModel,
#         generator::Generators.AbstractGenerator,
#         max_iter::Int = 100,
#         num_counterfactuals::Int = 1,
#         initialization::Symbol = :add_perturbation,
#         generative_model_params::NamedTuple = (;),
#         min_success_rate::AbstractFloat=0.99,
#     )

# Outer method to construct a `CounterfactualExplanation` structure.
# """
# function CounterfactualExplanation(
#     x::AbstractArray,
#     target::RawTargetType,
#     data::CounterfactualData,
#     M::Models.AbstractFittedModel,
#     generator::Generators.AbstractGenerator;
#     num_counterfactuals::Int=1,
#     initialization::Symbol=:add_perturbation,
#     generative_model_params::NamedTuple=(;),
#     max_iter::Int=100,
#     decision_threshold::AbstractFloat=0.5,
#     gradient_tol::AbstractFloat=parameters[:τ],
#     min_success_rate::AbstractFloat=parameters[:min_success_rate],
#     converge_when::Symbol=:decision_threshold,
# )

#     # Assertions:
#     @assert any(predict_label(M, data) .== target) "You model `M` never predicts the target value `target` for any of the samples contained in `data`. Are you sure the model is correctly specified?"
#     @assert 0.0 < min_success_rate <= 1.0 "Minimum success rate should be ∈ [0.0,1.0]."
#     @assert converge_when ∈ [:decision_threshold, :generator_conditions, :max_iter]
>>>>>>> 777e5003 (Broke up functions.jl)

#     # Factual:
#     x = typeof(x) == Int ? select_factual(data, x) : x

#     # Target:
#     target_encoded = data.output_encoder(target)

<<<<<<< HEAD
    # Initial Parameters:
    params = Dict{Symbol,Any}(
        :mutability => DataPreprocessing.mutability_constraints(data),
        :latent_space => generator.latent_space,
        :invalidation_rate => invalidation_rate,
        :learning_rate => learning_rate,
        :variance => variance,
    )
    ids = findall(predict_label(M, data) .== target)
    n_candidates = minimum([size(data.y, 2), 1000])
    candidates = select_factual(data, rand(ids, n_candidates))
    params[:potential_neighbours] = reduce(hcat, map(x -> x[1], collect(candidates)))
=======
#     # Initial Parameters:
#     params = Dict{Symbol,Any}(
#         :mutability => DataPreprocessing.mutability_constraints(data),
#         :latent_space => generator.latent_space,
#     )
#     ids = findall(predict_label(M, data) .== target)
#     n_candidates = minimum([size(data.y, 2), 1000])
#     candidates = select_factual(data, rand(ids, n_candidates))
#     params[:potential_neighbours] = reduce(hcat, map(x -> x[1], collect(candidates)))
>>>>>>> 777e5003 (Broke up functions.jl)

#     # Convergence Parameters:
#     convergence = Dict(
#         :max_iter => max_iter,
#         :decision_threshold => decision_threshold,
#         :gradient_tol => gradient_tol,
#         :min_success_rate => min_success_rate,
#         :converge_when => converge_when,
#     )

#     # Instantiate: 
#     ce = CounterfactualExplanation(
#         x,
#         target,
#         target_encoded,
#         x,
#         data,
#         M,
#         deepcopy(generator),
#         generative_model_params,
#         params,
#         nothing,
#         convergence,
#         num_counterfactuals,
#         initialization,
#     )

#     # Initialization:
#     adjust_shape!(ce)                                           # adjust shape to specified number of counterfactuals
#     ce.s′ = encode_state(ce)            # encode the counterfactual state
#     ce.s′ = initialize_state(ce)        # initialize the counterfactual state

#     # Initialize search:
#     ce.search = Dict(
#         :iteration_count => 0,
#         :times_changed_features => zeros(size(decode_state(ce))),
#         :path => [ce.s′],
#         :terminated => threshold_reached(ce, ce.x),
#         :converged => converged(ce),
#     )

#     # Check for redundancy:
#     if terminated(ce)
#         @info "Factual already in target class and probability exceeds threshold γ."
#     end

#     return ce
# end

<<<<<<< HEAD
# 1.) Convenience methods:
"""
	output_dim(ce::CounterfactualExplanation)
=======
# # 1.) Convenience methods:
# """
#     output_dim(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that returns the output dimension of the predictive model.
# """
# function output_dim(ce::CounterfactualExplanation)
#     return size(Models.probs(ce.M, ce.x))[1]
# end

<<<<<<< HEAD
"""
	guess_loss(ce::CounterfactualExplanation)
=======
# """
#     guess_loss(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Guesses the loss function to be used for the counterfactual search in case `likelihood` field is specified for the [`AbstractFittedModel`](@ref) instance and no loss function was explicitly declared for [`AbstractGenerator`](@ref) instance.
# """
# function guess_loss(ce::CounterfactualExplanation)
#     if :likelihood in fieldnames(typeof(ce.M))
#         if ce.M.likelihood == :classification_binary
#             loss_fun = Objectives.logitbinarycrossentropy
#         elseif ce.M.likelihood == :classification_multi
#             loss_fun = Objectives.logitcrossentropy
#         else
#             loss_fun = Flux.Losses.mse
#         end
#     else
#         loss_fun = nothing
#     end
#     return loss_fun
# end

<<<<<<< HEAD
# 2.) Initialisation
"""
	adjust_shape(
		ce::CounterfactualExplanation, 
		x::AbstractArray
	)
=======
# # 2.) Initialisation
# """
#     adjust_shape(
#         ce::CounterfactualExplanation, 
#         x::AbstractArray
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that adjusts the dimensions of `x`.
# """
# function adjust_shape(ce::CounterfactualExplanation, x::AbstractArray)
#     s′ = repeat(x; outer=(1, ce.num_counterfactuals))
#     return s′
# end

<<<<<<< HEAD
"""
	adjust_shape!(ce::CounterfactualExplanation)
=======
# """
#     adjust_shape!(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that adjusts the dimensions of the counterfactual state and related fields.
# """
# function adjust_shape!(ce::CounterfactualExplanation)

#     # Dimensionality:
#     x = deepcopy(ce.x)
#     s′ = adjust_shape(ce, x)      # augment to account for specified number of counterfactuals
#     ce.s′ = s′
#     target_encoded = ce.target_encoded
#     ce.target_encoded = adjust_shape(ce, target_encoded)

#     # Parameters:
#     params = ce.params
#     params[:mutability] = adjust_shape(ce, params[:mutability])      # augment to account for specified number of counterfactuals
#     return ce.params = params
# end

<<<<<<< HEAD
"""
	function encode_state(
		ce::CounterfactualExplanation, 
		x::Union{AbstractArray,Nothing} = nothing,
	)
=======
# """
#     function encode_state(
#         ce::CounterfactualExplanation, 
#         x::Union{AbstractArray,Nothing} = nothing,
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# Applies all required encodings to `x`:

# 1. If applicable, it maps `x` to the latent space learned by the generative model.
# 2. If and where applicable, it rescales features. 

# Finally, it returns the encoded state variable.
# """
# function encode_state(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )

#     # Unpack:
#     s′ = isnothing(x) ? deepcopy(ce.s′) : x
#     data = ce.data

#     # Latent space:
#     if ce.params[:latent_space]
#         s′ = map_to_latent(ce, s′)
#     end

#     # Standardize data unless latent space:
#     if !ce.params[:latent_space] && data.standardize
#         dt = data.dt
#         idx = transformable_features(data)
#         ignore_derivatives() do
#             s = s′[idx, :]
#             StatsBase.transform!(dt, s)
#             s′[idx, :] = s
#         end
#     end

#     return s′
# end

<<<<<<< HEAD
"""
	wants_latent_space(
		ce::CounterfactualExplanation, 
		x::Union{AbstractArray,Nothing} = nothing,
	)   
=======
# """
#     wants_latent_space(
#         ce::CounterfactualExplanation, 
#         x::Union{AbstractArray,Nothing} = nothing,
#     )   
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience function that checks if latent space search is applicable.
# """
# function wants_latent_space(ce::CounterfactualExplanation)

#     # Unpack:
#     latent_space = ce.params[:latent_space]

#     # If threshold is already reached, training GM is redundant:
#     latent_space = latent_space && !threshold_reached(ce, ce.x)

#     return latent_space
# end

<<<<<<< HEAD
@doc raw"""
   function map_to_latent(
		ce::CounterfactualExplanation,
		x::Union{AbstractArray,Nothing}=nothing,
	) 
=======
# @doc raw"""
#    function map_to_latent(
#         ce::CounterfactualExplanation,
#         x::Union{AbstractArray,Nothing}=nothing,
#     ) 
>>>>>>> 777e5003 (Broke up functions.jl)

# Maps `x` from the feature space $\mathcal{X}$ to the latent space learned by the generative model.
# """
# function map_to_latent(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )

#     # Unpack:
#     s′ = isnothing(x) ? deepcopy(ce.s′) : x
#     data = ce.data
#     generator = ce.generator

#     if ce.params[:latent_space]
#         @info "Searching in latent space using generative model."
#         generative_model = DataPreprocessing.get_generative_model(
#             data; ce.generative_model_params...
#         )
#         # map counterfactual to latent space: s′=z′∼p(z|x)
#         s′, _, _ = GenerativeModels.rand(generative_model.encoder, s′)
#     end

#     return s′
# end

<<<<<<< HEAD
"""
	function decode_state(
		ce::CounterfactualExplanation,
		x::Union{AbstractArray,Nothing}=nothing,
	)
=======
# """
#     function decode_state(
#         ce::CounterfactualExplanation,
#         x::Union{AbstractArray,Nothing}=nothing,
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# Applies all the applicable decoding functions:

# 1. If applicable, map the state variable back from the latent space to the feature space.
# 2. If and where applicable, inverse-transform features.
# 3. Reconstruct all categorical encodings.

# Finally, the decoded counterfactual is returned.
# """
# function decode_state(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )

#     # Unpack:
#     s′ = isnothing(x) ? deepcopy(ce.s′) : x
#     data = ce.data

#     # Latent space:
#     if ce.params[:latent_space]
#         s′ = map_from_latent(ce, s′)
#     end

#     # Standardization:
#     if !ce.params[:latent_space] && data.standardize
#         dt = data.dt

#         # Continuous:
#         idx = transformable_features(data)
#         ignore_derivatives() do
#             s = s′[idx, :]
#             StatsBase.reconstruct!(dt, s)
#             s′[idx, :] = s
#         end
#     end

#     # Categorical:
#     s′ = reconstruct_cat_encoding(ce, s′)

#     return s′
# end

<<<<<<< HEAD
"""
	map_from_latent(
		ce::CounterfactualExplanation,
		x::Union{AbstractArray,Nothing}=nothing,
	)
=======
# """
#     map_from_latent(
#         ce::CounterfactualExplanation,
#         x::Union{AbstractArray,Nothing}=nothing,
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# Maps the state variable back from the latent space to the feature space.
# """
# function map_from_latent(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )

#     # Unpack:
#     s′ = isnothing(x) ? deepcopy(ce.s′) : x
#     data = ce.data

#     # Latent space:
#     if ce.params[:latent_space]
#         generative_model = data.generative_model
#         if !isnothing(generative_model)
#             # NOTE! This is not very clean, will be improved.
#             if generative_model.params.nll == Flux.Losses.logitbinarycrossentropy
#                 s′ = Flux.σ.(generative_model.decoder(s′))
#             else
#                 s′ = generative_model.decoder(s′)
#             end
#         end
#     end

#     return s′
# end

<<<<<<< HEAD
"""
	reconstruct_cat_encoding(
		ce::CounterfactualExplanation,
		x::Union{AbstractArray,Nothing}=nothing,
	)
=======
# """
#     reconstruct_cat_encoding(
#         ce::CounterfactualExplanation,
#         x::Union{AbstractArray,Nothing}=nothing,
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# Reconstructs all categorical encodings. See [`DataPreprocessing.reconstruct_cat_encoding`](@ref) for details.
# """
# function reconstruct_cat_encoding(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )
#     # Unpack:
#     s′ = isnothing(x) ? deepcopy(ce.s′) : x
#     data = ce.data

#     s′ = DataPreprocessing.reconstruct_cat_encoding(data, s′)

#     return s′
# end

<<<<<<< HEAD
"""
	initialize_state(ce::CounterfactualExplanation)

Initializes the starting point for the factual(s):
	
1. If `ce.initialization` is set to `:identity` or counterfactuals are searched in a latent space, then nothing is done.
2. If `ce.initialization` is set to `:add_perturbation`, then a random perturbation is added to the factual following following Slack (2021): https://arxiv.org/abs/2106.02666. The authors show that this improves adversarial robustness.
"""
function initialize_state(ce::CounterfactualExplanation)
    @assert ce.initialization ∈ [:identity, :add_perturbation]
=======
# """
#     initialize_state(ce::CounterfactualExplanation)

# Initializes the starting point for the factual(s):
    
# 1. If `ce.initialization` is set to `:identity` or counterfactuals are searched in a latent space, then nothing is done.
# 2. If `ce.initialization` is set to `:add_perturbation`, then a random perturbation is added to the factual following following Slack (2021): https://arxiv.org/abs/2106.02666. The authors show that this improves adversarial robustness.
# """
# function initialize_state(ce::CounterfactualExplanation)
#     @assert ce.initialization ∈ [:identity, :add_perturbation]
>>>>>>> 777e5003 (Broke up functions.jl)

#     s′ = ce.s′
#     data = ce.data

#     # No perturbation:
#     if ce.initialization == :identity
#         return s′
#     end

#     # If latent space, initial point is random anyway:
#     if ce.params[:latent_space]
#         return s′
#     end

#     # Add random perturbation following Slack (2021): https://arxiv.org/abs/2106.02666
#     if ce.initialization == :add_perturbation
#         Δs′ = randn(eltype(s′), size(s′)) * convert(eltype(s′), 0.1)
#         Δs′ = apply_mutability(ce, Δs′)
#         s′ .+= Δs′
#     end

#     return s′
# end

<<<<<<< HEAD
# 3.) Information about CE
"""
	factual(ce::CounterfactualExplanation)
=======
# # 3.) Information about CE
# """
#     factual(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method to retrieve the factual `x`.
# """
# function factual(ce::CounterfactualExplanation)
#     return ce.x
# end

<<<<<<< HEAD
"""
	factual_probability(ce::CounterfactualExplanation)
=======
# """
#     factual_probability(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method to compute the class probabilities of the factual.
# """
# function factual_probability(ce::CounterfactualExplanation)
#     return Models.probs(ce.M, ce.x)
# end

<<<<<<< HEAD
"""
	factual_label(ce::CounterfactualExplanation)  
=======
# """
#     factual_label(ce::CounterfactualExplanation)  
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method to get the predicted label associated with the factual.
# """
# function factual_label(ce::CounterfactualExplanation)
#     M = ce.M
#     counterfactual_data = ce.data
#     y = predict_label(M, counterfactual_data, factual(ce))
#     return y
# end

<<<<<<< HEAD
"""
	counterfactual(ce::CounterfactualExplanation)
=======
# """
#     counterfactual(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that returns the counterfactual.
# """
# function counterfactual(ce::CounterfactualExplanation)
#     return decode_state(ce)
# end

<<<<<<< HEAD
"""
	counterfactual_probability(ce::CounterfactualExplanation)
=======
# """
#     counterfactual_probability(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that computes the class probabilities of the counterfactual.
# """
# function counterfactual_probability(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )
#     if isnothing(x)
#         x = counterfactual(ce)
#     end
#     p = Models.probs(ce.M, x)
#     return p
# end

<<<<<<< HEAD
"""
	counterfactual_label(ce::CounterfactualExplanation) 
=======
# """
#     counterfactual_label(ce::CounterfactualExplanation) 
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that returns the predicted label of the counterfactual.
# """
# function counterfactual_label(ce::CounterfactualExplanation)
#     M = ce.M
#     counterfactual_data = ce.data
#     y = predict_label(M, counterfactual_data, counterfactual(ce))
#     return y
# end

<<<<<<< HEAD
"""
	target_probs(
		ce::CounterfactualExplanation,
		x::Union{AbstractArray,Nothing}=nothing,
	)
=======
# """
#     target_probs(
#         ce::CounterfactualExplanation,
#         x::Union{AbstractArray,Nothing}=nothing,
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# Returns the predicted probability of the target class for `x`. If `x` is `nothing`, the predicted probability corresponding to the counterfactual value is returned.
# """
# function target_probs(
#     ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
# )
#     data = ce.data
#     likelihood = ce.data.likelihood
#     p = counterfactual_probability(ce, x)
#     target = ce.target
#     target_idx = get_target_index(data.y_levels, target)
#     if likelihood == :classification_binary
#         if target_idx == 2
#             p_target = p
#         else
#             p_target = 1 .- p
#         end
#     else
#         p_target = selectdim(p, 1, target_idx)
#     end
#     return p_target
# end

<<<<<<< HEAD
# 4.) Search related methods:
"""
	path(ce::CounterfactualExplanation)
=======
# # 4.) Search related methods:
# """
#     path(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that returns the entire counterfactual path.
# """
# function path(ce::CounterfactualExplanation; feature_space=true)
#     path = deepcopy(ce.search[:path])
#     if feature_space
#         path = [decode_state(ce, z) for z in path]
#     end
#     return path
# end

<<<<<<< HEAD
"""
	counterfactual_probability_path(ce::CounterfactualExplanation)
=======
# """
#     counterfactual_probability_path(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Returns the counterfactual probabilities for each step of the search.
# """
# function counterfactual_probability_path(ce::CounterfactualExplanation)
#     M = ce.M
#     p = map(X -> counterfactual_probability(ce, X), path(ce))
#     return p
# end

<<<<<<< HEAD
"""
	counterfactual_label_path(ce::CounterfactualExplanation)
=======
# """
#     counterfactual_label_path(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Returns the counterfactual labels for each step of the search.
# """
# function counterfactual_label_path(ce::CounterfactualExplanation)
#     counterfactual_data = ce.data
#     M = ce.M
#     ŷ = map(X -> predict_label(M, counterfactual_data, X), path(ce))
#     return ŷ
# end

<<<<<<< HEAD
"""
	target_probs_path(ce::CounterfactualExplanation)
=======
# """
#     target_probs_path(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Returns the target probabilities for each step of the search.
# """
# function target_probs_path(ce::CounterfactualExplanation)
#     X = path(ce)
#     P = map(x -> target_probs(ce, x), X)
#     return P
# end

<<<<<<< HEAD
"""
	embed_path(ce::CounterfactualExplanation)
=======
# """
#     embed_path(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Helper function that embeds path into two dimensions for plotting.
# """
# function embed_path(ce::CounterfactualExplanation)
#     data_ = ce.data
#     return DataPreprocessing.embed(data_, path(ce))
# end

<<<<<<< HEAD
"""
	apply_mutability(
		ce::CounterfactualExplanation,
		Δs′::AbstractArray,
	)
=======
# """
#     apply_mutability(
#         ce::CounterfactualExplanation,
#         Δs′::AbstractArray,
#     )
>>>>>>> 777e5003 (Broke up functions.jl)

# A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
# """
# function apply_mutability(ce::CounterfactualExplanation, Δs′::AbstractArray)
#     if ce.params[:latent_space]
#         if isnothing(ce.search)
#             @warn "Mutability constraints not currently implemented for latent space search."
#         end
#         return Δs′
#     end

#     mutability = ce.params[:mutability]
#     # Helper functions:
#     both(x) = x
#     increase(x) = ifelse(x < 0.0, 0.0, x)
#     decrease(x) = ifelse(x > 0.0, 0.0, x)
#     none(x) = 0.0
#     cases = (both=both, increase=increase, decrease=decrease, none=none)

#     # Apply:
#     Δs′ = map((case, s) -> getfield(cases, case)(s), mutability, Δs′)

#     return Δs′
# end

<<<<<<< HEAD
"""
	apply_domain_constraints!(ce::CounterfactualExplanation)
=======
# """
#     apply_domain_constraints!(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Wrapper function that applies underlying domain constraints.
# """
# function apply_domain_constraints!(ce::CounterfactualExplanation)
#     if !wants_latent_space(ce)
#         s′ = ce.s′
#         ce.s′ = DataPreprocessing.apply_domain_constraints(ce.data, s′)
#     end
# end

<<<<<<< HEAD
# 5.) Convergence related methods:
"""
	terminated(ce::CounterfactualExplanation)
=======
# # 5.) Convergence related methods:
# """
#     terminated(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method to determine if the counterfactual search has terminated.
# """
# function terminated(ce::CounterfactualExplanation)
#     return converged(ce) || steps_exhausted(ce)
# end

<<<<<<< HEAD
"""
	converged(ce::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has converged. The search is considered to have converged only if the counterfactual is valid.
"""
function converged(ce::CounterfactualExplanation)
    if ce.convergence[:converge_when] == :decision_threshold
        conv = threshold_reached(ce)
    elseif ce.convergence[:converge_when] == :generator_conditions
        conv = threshold_reached(ce) && Generators.conditions_satisfied(ce.generator, ce)
    elseif ce.convergence[:converge_when] == :max_iter
        conv = false
    elseif ce.convergence[:converge_when] == :invalidation_rate
        ir = Generators.invalidation_rate(ce)
        # gets the label from an array, not sure why it is an array though.
        label = predict_label(ce.M, ce.data, decode_state(ce))[1]
        conv = label == ce.target && ce.params[:invalidation_rate] > ir
    else
        @error "Convergence criterion not recognized."
    end
=======
# """
#     converged(ce::CounterfactualExplanation)

# A convenience method to determine if the counterfactual search has converged. The search is considered to have converged only if the counterfactual is valid.
# """
# function converged(ce::CounterfactualExplanation)
#     if ce.convergence[:converge_when] == :decision_threshold
#         conv = threshold_reached(ce)
#     elseif ce.convergence[:converge_when] == :generator_conditions
#         conv = threshold_reached(ce) && Generators.conditions_satisfied(ce.generator, ce)
#     elseif ce.convergence[:converge_when] == :max_iter
#         conv = false
#     end
>>>>>>> 777e5003 (Broke up functions.jl)

#     return conv
# end

<<<<<<< HEAD
"""
	threshold_reached(ce::CounterfactualExplanation)
=======
# """
#     threshold_reached(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that determines if the predefined threshold for the target class probability has been reached.
# """
# function threshold_reached(ce::CounterfactualExplanation)
#     γ = ce.convergence[:decision_threshold]
#     success_rate = sum(target_probs(ce) .>= γ) / ce.num_counterfactuals
#     return success_rate > ce.convergence[:min_success_rate]
# end

<<<<<<< HEAD
"""
	threshold_reached(ce::CounterfactualExplanation, x::AbstractArray)
=======
# """
#     threshold_reached(ce::CounterfactualExplanation, x::AbstractArray)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that determines if the predefined threshold for the target class probability has been reached for a specific sample `x`.
# """
# function threshold_reached(ce::CounterfactualExplanation, x::AbstractArray)
#     γ = ce.convergence[:decision_threshold]
#     success_rate = sum(target_probs(ce, x) .>= γ) / ce.num_counterfactuals
#     return success_rate > ce.convergence[:min_success_rate]
# end

<<<<<<< HEAD
"""
	steps_exhausted(ce::CounterfactualExplanation) 
=======
# """
#     steps_exhausted(ce::CounterfactualExplanation) 
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that checks if the number of maximum iterations has been exhausted.
# """
# function steps_exhausted(ce::CounterfactualExplanation)
#     return ce.search[:iteration_count] == ce.convergence[:max_iter]
# end

<<<<<<< HEAD
"""
	total_steps(ce::CounterfactualExplanation)
=======
# """
#     total_steps(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# A convenience method that returns the total number of steps of the counterfactual search.
# """
# function total_steps(ce::CounterfactualExplanation)
#     return ce.search[:iteration_count]
# end

<<<<<<< HEAD
# UPDATES
"""
	update!(ce::CounterfactualExplanation) 

An important subroutine that updates the counterfactual explanation.
It takes a snapshot of the current counterfactual search state and passes it to the generator.
Based on the current state the generator generates perturbations.
Various constraints are then applied to the proposed vector of feature perturbations.
Finally, the counterfactual search state is updated.
"""
function update!(ce::CounterfactualExplanation)

    # Generate peturbations:
    Δs′ = Generators.generate_perturbations(ce.generator, ce)
    Δs′ = apply_mutability(ce, Δs′)         # mutability constraints
    s′ = ce.s′ + ce.params[:learning_rate] .* Δs′                         # new proposed state
=======
# # UPDATES
# """
#     update!(ce::CounterfactualExplanation) 

# An important subroutine that updates the counterfactual explanation. It takes a snapshot of the current counterfactual search state and passes it to the generator. Based on the current state the generator generates perturbations. Various constraints are then applied to the proposed vector of feature perturbations. Finally, the counterfactual search state is updated.
# """
# function update!(ce::CounterfactualExplanation)

#     # Generate peturbations:
#     Δs′ = Generators.generate_perturbations(ce.generator, ce)
#     Δs′ = apply_mutability(ce, Δs′)         # mutability constraints
#     s′ = ce.s′ + Δs′                        # new proposed state
>>>>>>> 777e5003 (Broke up functions.jl)

#     # Updates:
#     ce.s′ = s′                                                  # update counterfactual
#     _times_changed = reshape(
#         decode_state(ce, Δs′) .!= 0, size(ce.search[:times_changed_features])
#     )
#     ce.search[:times_changed_features] += _times_changed        # update number of times feature has been changed
#     ce.search[:mutability] = Generators.mutability_constraints(ce.generator, ce)
#     ce.search[:iteration_count] += 1                            # update iteration counter   
#     ce.search[:path] = [ce.search[:path]..., ce.s′]
#     ce.search[:converged] = converged(ce)
#     return ce.search[:terminated] = terminated(ce)
# end

<<<<<<< HEAD
"""
	get_meta(ce::CounterfactualExplanation)
=======
# """
#     get_meta(ce::CounterfactualExplanation)
>>>>>>> 777e5003 (Broke up functions.jl)

# Returns meta data for a counterfactual explanation.
# """
# function get_meta(ce::CounterfactualExplanation)
#     meta_data = Dict(:model => Symbol(ce.M), :generator => Symbol(ce.generator))
#     return meta_data
# end

# function Base.show(io::IO, z::CounterfactualExplanation)
#     println(io, "")
#     if z.search[:iteration_count] > 0
#         if isnothing(z.convergence[:decision_threshold])
#             p_path = target_probs_path(z)
#             n_reached = findall([
#                 all(p .>= z.convergence[:decision_threshold]) for p in p_path
#             ])
#             if length(n_reached) > 0
#                 printstyled(
#                     io,
#                     "Threshold reached: $(all(threshold_reached(z)) ? "✅"  : "❌")";
#                     bold=true,
#                 )
#                 print(" after $(first(n_reached)) steps.\n")
#             end
#             printstyled(io, "Convergence: $(converged(z) ? "✅"  : "❌")"; bold=true)
#             print(" after $(total_steps(z)) steps.\n")
#         else
#             printstyled(io, "Convergence: $(converged(z) ? "✅"  : "❌")"; bold=true)
#             print(" after $(total_steps(z)) steps.\n")
#         end
#     end
# end
