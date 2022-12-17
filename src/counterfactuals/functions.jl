using ChainRulesCore: @ignore_derivatives
using Flux
using MLUtils
using SliceMap
using Statistics
using StatsBase

"""
A struct that collects all information relevant to a specific counterfactual explanations for a single individual.
"""
mutable struct CounterfactualExplanation <: AbstractCounterfactualExplanation
    x::AbstractArray
    target::Number
    target_encoded::Union{Number, AbstractArray, Nothing}
    s′::AbstractArray
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractFittedModel
    generator::Generators.AbstractGenerator
    latent_space::Union{Nothing, Bool}
    generative_model_params::NamedTuple
    params::Dict
    search::Union{Dict,Nothing}
    num_counterfactuals::Int
    initialization::Symbol
end

"""
    function CounterfactualExplanation(
        ;
        x::AbstractArray, 
        target::Union{AbstractFloat,Int}, 
        data::CounterfactualData,  
        M::Models.AbstractFittedModel,
        generator::Generators.AbstractGenerator,
        T::Int=100,
        latent_space::Union{Nothing, Bool}=nothing,
        num_counterfactuals::Int=1,
        initialization::Symbol=:add_perturbation,
        generative_model_params::NamedTuple=(;)
    ) 

Outer method to construct a `CounterfactualExplanation` structure.
"""
function CounterfactualExplanation(
    ;
    x::AbstractArray, 
    target::Union{AbstractFloat,Int}, 
    data::CounterfactualData,  
    M::Models.AbstractFittedModel,
    generator::Generators.AbstractGenerator,
    T::Int=100,
    latent_space::Union{Nothing, Bool}=nothing,
    num_counterfactuals::Int=1,
    initialization::Symbol=:add_perturbation,
    generative_model_params::NamedTuple=(;)
) 

    # Factual:
    x = typeof(x) == Int ? select_factual(data, x) : x

    # Initial Parameters:
    params = Dict(
        :γ => isnothing(generator.decision_threshold) ? 0.5 : generator.decision_threshold,
        :T => T,
        :mutability => DataPreprocessing.mutability_constraints(data),
        :initial_mutability => DataPreprocessing.mutability_constraints(data),
    )
    ids = getindex.(findall(data.y.==target),2)
    n_candidates = minimum([size(data.y,2),1000])
    candidates = select_factual(data,rand(ids,n_candidates))
    params[:potential_neighbours] = reduce(hcat, map(x -> x[1], collect(candidates)))

    # Instantiate: 
    counterfactual_explanation = CounterfactualExplanation(
        x, target, nothing, x,
        data, M, generator, latent_space, generative_model_params, 
        params, nothing, num_counterfactuals, initialization
    )

    # Initialization:
    reshape!(counterfactual_explanation)                                                        # adjust shape to specified number of counterfactuals
    counterfactual_explanation.s′ = encode_state(counterfactual_explanation)                    # encode the counterfactual state
    counterfactual_explanation.s′ = initialize_state(counterfactual_explanation)                # initialize the counterfactual state
    counterfactual_explanation.target_encoded = encode_target(counterfactual_explanation)       # encode the target variable

    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 0,
        :times_changed_features => zeros(size(decode_state(counterfactual_explanation))),
        :path => [counterfactual_explanation.s′],
        :terminated => threshold_reached(counterfactual_explanation, counterfactual_explanation.x),
        :converged => converged(counterfactual_explanation),
    )

    # Check for redundancy:
    if terminated(counterfactual_explanation)
        @info "Factual already in target class and probability exceeds threshold γ."
    end

    return counterfactual_explanation

end

# Convenience methods:
"""
    output_dim(counterfactual_explanation::CounterfactualExplanation)

A convenience method that computes the output dimension of the predictive model.
"""
output_dim(counterfactual_explanation::CounterfactualExplanation) = size(Models.probs(counterfactual_explanation.M, counterfactual_explanation.x))[1]

function reshape!(counterfactual_explanation::CounterfactualExplanation)
    # Dimensionality:
    x = counterfactual_explanation.x
    size_ = Int.(vcat(ones(maximum([ndims(x),2])),counterfactual_explanation.num_counterfactuals))
    s′ = copy(x)                                        # start from factual
    s′ = repeat(x, outer=size_)                         # augment to account for specified number of counterfactuals
    counterfactual_explanation.s′ = s′
    
    # Parameters:
    params = counterfactual_explanation.params
    params[:mutability] = repeat(params[:mutability], outer=size_)
    params[:initial_mutability] = params[:mutability]
    counterfactual_explanation.params = params
end

"""
    encode_state(counterfactual_explanations::CounterfactualExplanation)

Encodes counterfactual.
"""
function encode_state(counterfactual_explanation::CounterfactualExplanation)

    s′ = counterfactual_explanation.s′
    data = counterfactual_explanation.data
    generator = counterfactual_explanation.generator

    # Standardization:
    # dt = data.dt
    # features_continuous = data.features_continuous
    # s′ = SliceMap.slicemap(s′, dims=(1,2)) do s
    #     s[features_continuous,:] = StatsBase.transform(dt, s[features_continuous,:])
    #     return s
    # end

    # Categorical encoding:
    # --------------------- #

    # Latent space:
    latent_space = counterfactual_explanation.latent_space
    wants_latent_space = DataPreprocessing.has_pretrained_generative_model(data) || typeof(generator) <: Generators.AbstractLatentSpaceGenerator
    # Only overwrite, if not explicitly set to false:
    counterfactual_explanation.latent_space = isnothing(latent_space) ? wants_latent_space : latent_space
    if counterfactual_explanation.latent_space && !threshold_reached(counterfactual_explanation, counterfactual_explanation.x)
        @info "Searching in latent space using generative model."
        generative_model = DataPreprocessing.get_generative_model(data; counterfactual_explanation.generative_model_params...)
        # map counterfactual to latent space: s′=z′∼p(z|x)
        s′, _, _ = GenerativeModels.rand(generative_model.encoder, s′)
    end

    return s′

end

function decode_state(counterfactual_explanation::CounterfactualExplanation, x::Union{AbstractArray, Nothing}=nothing)
    
    s′ = isnothing(x) ? counterfactual_explanation.s′ : x

    # Standardization:
    # dt = counterfactual_explanation.data.dt
    # features_continuous = counterfactual_explanation.data.features_continuous
    # s′ = mapslices(s -> StatsBase.reconstruct(dt, s, features_continuous), s′, dims=(1,2)) 
    
    # Categorical encoding:
    # --------------------- #

    # Latent space:
    if counterfactual_explanation.latent_space
        generative_model = counterfactual_explanation.data.generative_model

        # NOTE! This is not very clean, will be improved.
        if generative_model.params.nll==Flux.Losses.logitbinarycrossentropy
            s′ = Flux.σ.(generative_model.decoder(s′))
        else
            s′ = generative_model.decoder(s′)
        end
    end

    return s′

end

"""
    encode_target(counterfactual_explanation::CounterfactualExplanation) 

A convenience method to encode the target variable, if necessary.
"""
function encode_target(counterfactual_explanation::CounterfactualExplanation) 
    out_dim = output_dim(counterfactual_explanation)
    target = counterfactual_explanation.target
    target = out_dim > 1 ? Flux.onehot(target, 1:out_dim) : [target]
    target = repeat(target, outer=[1,1,counterfactual_explanation.num_counterfactuals])
    return target
end

"""
    initialize_state(counterfactual_explanation::CounterfactualExplanation)

Initializes the starting point for the factual(s).
"""
function initialize_state(counterfactual_explanation::CounterfactualExplanation)

    @assert counterfactual_explanation.initialization ∈ [:identity, :add_perturbation]  

    s′ = counterfactual_explanation.s′
    data = counterfactual_explanation.data

    if counterfactual_explanation.initialization == :add_perturbation
        scale = std(data.X, dims=2) .* 0.1
        s′ = SliceMap.slicemap(s′, dims=(1,2)) do s
            Δs′ = scale .* randn(size(scale,1))
            Δs′ = apply_mutability(counterfactual_explanation, Δs′)    
            s .+ Δs′
        end
    end

<<<<<<< HEAD
    return s′

=======
>>>>>>> main
end

# 1) Factual values
"""
    factual(counterfactual_explanation::CounterfactualExplanation)

A convenience method to get the factual value.
"""
factual(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.x

"""
    factual_probability(counterfactual_explanation::CounterfactualExplanation)

A convenience method to compute the class probabilities of the factual value.
"""
factual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.M, counterfactual_explanation.x)

"""
    factual_label(counterfactual_explanation::CounterfactualExplanation)  

A convenience method to get the predicted label associated with the factual value.
"""
function factual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = factual_probability(counterfactual_explanation)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end

# 2) Counterfactual values:
"""
    counterfactual(counterfactual_explanation::CounterfactualExplanation)

A convenience method to get the counterfactual value.
"""
counterfactual(counterfactual_explanation::CounterfactualExplanation) = decode_state(counterfactual_explanation)

"""
    counterfactual_probability(counterfactual_explanation::CounterfactualExplanation)

A convenience method to compute the class probabilities of the counterfactual value.
"""
counterfactual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.M, counterfactual(counterfactual_explanation))

"""
    _to_label(p::AbstractArray)

Small helper function mapping predicted probabilities to labels.
"""
function _to_label(p::AbstractArray)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end 

"""
    counterfactual_label(counterfactual_explanation::CounterfactualExplanation) 

A convenience method to get the predicted label associated with the counterfactual value.
"""
function counterfactual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = counterfactual_probability(counterfactual_explanation)
    y = SliceMap.slicemap(p -> _to_label(p), p, dims=(1,2))
    return y
end

"""
    target_probs(counterfactual_explanation::CounterfactualExplanation, x::Union{AbstractArray, Nothing}=nothing)

Returns the predicted probability of the target class for `x`. If `x` is `nothing`, the predicted probability corresponding to the counterfactual value is returned.
"""
function target_probs(counterfactual_explanation::CounterfactualExplanation, x::Union{AbstractArray, Nothing}=nothing)
    
    p = !isnothing(x) ? Models.probs(counterfactual_explanation.M, x) : counterfactual_probability(counterfactual_explanation)
    target = counterfactual_explanation.target

    if size(p,1) == 1
        h(x) = ifelse(x==-1,0,x)
        if target ∉ [0,1] && target ∉ [-1,1]
            throw(DomainError("For binary classification expecting target to be in {0,1} or {-1,1}.")) 
        end
        # If target is binary (i.e. outcome 1D from sigmoid), compute p(y=0):
        p = vcat(1.0 .- p, p)
        # Choose first (target+1) row if target=0, second row (target+1) if target=1:  
        p_target = selectdim(p,1,Int(h(target)+1))
    else
        if target < 1 || target % 1 !=0
            throw(DomainError("For multi-class classification expecting `target` ∈ ℕ⁺, i.e. {1,2,3,...}.")) 
        end
        # If target is multi-class, choose corresponding row (e.g. target=2 -> row 2)
        p_target = selectdim(p,1,Int(target)) 
    end
    return p_target
end

# 3) Search related methods:
"""
    terminated(counterfactual_explanation::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has terminated.
"""
function terminated(counterfactual_explanation::CounterfactualExplanation)
    converged(counterfactual_explanation) || steps_exhausted(counterfactual_explanation)
end

"""
    converged(counterfactual_explanation::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has converged.
"""
function converged(counterfactual_explanation::CounterfactualExplanation)
    # If strict, also look at gradient and other generator-specific conditions.
    # Otherwise only check if probability threshold has been reached.
    if isnothing(counterfactual_explanation.generator.decision_threshold)
        threshold_reached(counterfactual_explanation) && Generators.conditions_satisified(counterfactual_explanation.generator, counterfactual_explanation)
    else
        threshold_reached(counterfactual_explanation)
    end
end

"""
    total_steps(counterfactual_explanation::CounterfactualExplanation)

A convenience method that returns the total number of steps of the counterfactual search.
"""
total_steps(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:iteration_count]

"""
    path(counterfactual_explanation::CounterfactualExplanation)

A convenience method that returns the entire counterfactual path.
"""
function path(counterfactual_explanation::CounterfactualExplanation; feature_space=true)
    path = counterfactual_explanation.search[:path]
    if feature_space
        path = [decode_state(counterfactual_explanation, z) for z ∈ path]
    end
    return path
end

"""
    counterfactual_probability_path(counterfactual_explanation::CounterfactualExplanation)

Returns the counterfactual probabilities for each step of the search.
"""
function counterfactual_probability_path(counterfactual_explanation::CounterfactualExplanation)
    M = counterfactual_explanation.M
    p = map(X -> mapslices(x -> probs(M, x), X, dims=(1,2)), path(counterfactual_explanation))
    return p
end

"""
    counterfactual_label_path(counterfactual_explanation::CounterfactualExplanation)

Returns the counterfactual labels for each step of the search.
"""
function counterfactual_label_path(counterfactual_explanation::CounterfactualExplanation)
    P = counterfactual_probability_path(counterfactual_explanation)
    ŷ = map(P -> mapslices(p -> _to_label(p), P, dims=(1,2)), P)
    return ŷ
end

"""
    target_probs_path(counterfactual_explanation::CounterfactualExplanation)

Returns the target probabilities for each step of the search.
"""
function target_probs_path(counterfactual_explanation::CounterfactualExplanation)
    X = path(counterfactual_explanation)
    P = map(X -> mapslices(x -> target_probs(counterfactual_explanation, x), X, dims=(1,2)), X)
    return P
end

"""
    embed_path(counterfactual_explanation::CounterfactualExplanation)

Helper function that embeds path into two dimensions for plotting.
"""
function embed_path(counterfactual_explanation::CounterfactualExplanation)
    data_ = counterfactual_explanation.data
    path_ = MLUtils.stack(path(counterfactual_explanation); dims = 1)
    path_embedded = mapslices(X -> DataPreprocessing.embed(data_, X'), path_, dims=(1,2))
    path_embedded = unstack(path_embedded,dims=2)
    return path_embedded
end

"""
    apply_mutability(counterfactual_explanation::CounterfactualExplanation, Δs′::AbstractArray)

A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
"""
function apply_mutability(counterfactual_explanation::CounterfactualExplanation, Δs′::AbstractArray)

    if counterfactual_explanation.latent_space
        if !all(counterfactual_explanation.params[:mutability].==:both) && total_steps(counterfactual_explanation) == 0
            @error "Mutability constraints not currently implemented for latent space search."
        end
    end

    mutability = counterfactual_explanation.params[:mutability]
    # Helper functions:
    both(x) = x
    increase(x) = ifelse(x<0.0,0.0,x)
    decrease(x) = ifelse(x>0.0,0.0,x)
    none(x) = 0.0
    cases = (both = both, increase = increase, decrease = decrease, none = none)

    # Apply:
    Δs′ = map((case,s) -> getfield(cases,case)(s),mutability,Δs′)

    return Δs′

end

"""
    apply_domain_constraints!(counterfactual_explanation::CounterfactualExplanation)

Wrapper function that applies underlying domain constraints.
"""
function apply_domain_constraints!(counterfactual_explanation::CounterfactualExplanation)

    if !isnothing(counterfactual_explanation.data.domain) && total_steps(counterfactual_explanation) == 0
        @error "Domain constraints not currently implemented for latent space search."
    end

    s′ = counterfactual_explanation.s′
    counterfactual_explanation.s′ = DataPreprocessing.apply_domain_constraints(counterfactual_explanation.data, s′)

end

"""
    threshold_reached(counterfactual_explanation::CounterfactualExplanation)

A convenience method that determines if the predefined threshold for the target class probability has been reached.
"""
function threshold_reached(counterfactual_explanation::CounterfactualExplanation)
    γ = isnothing(counterfactual_explanation.generator.decision_threshold) ? 0.5 : counterfactual_explanation.generator.decision_threshold
    all(target_probs(counterfactual_explanation) .>= γ)
end

"""
    threshold_reached(counterfactual_explanation::CounterfactualExplanation, x::AbstractArray)

A convenience method that determines if the predefined threshold for the target class probability has been reached for a specific sample `x`.
"""
function threshold_reached(counterfactual_explanation::CounterfactualExplanation, x::AbstractArray)
    γ = isnothing(counterfactual_explanation.generator.decision_threshold) ? 0.5 : counterfactual_explanation.generator.decision_threshold
    all(target_probs(counterfactual_explanation, x) .>= γ)
end

"""
    steps_exhausted(counterfactual_explanation::CounterfactualExplanation) 

A convenience method that checks if the number of maximum iterations has been exhausted.
"""
steps_exhausted(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:iteration_count] == counterfactual_explanation.params[:T]

"""
    guess_loss(counterfactual_explanation::CounterfactualExplanation)

Guesses the loss function to be used for the counterfactual search in case `likelihood` field is specified for the [`AbstractFittedModel`](@ref) instance and no loss function was explicitly declared for [`AbstractGenerator`](@ref) instance.
"""
function guess_loss(counterfactual_explanation::CounterfactualExplanation)
    if :likelihood in fieldnames(typeof(counterfactual_explanation.M))
        if counterfactual_explanation.M.likelihood == :classification_binary
            loss_fun = Flux.Losses.logitbinarycrossentropy
        elseif counterfactual_explanation.M.likelihood == :classification_multi
            loss_fun = Flux.Losses.logitcrossentropy
        else
            loss_fun = Flux.Losses.mse
        end
    else
        loss_fun = nothing
    end
    return loss_fun
end

"""
    update!(counterfactual_explanation::CounterfactualExplanation) 

An important subroutine that updates the counterfactual explanation. It takes a snapshot of the current counterfactual search state and passes it to the generator. Based on the current state the generator generates perturbations. Various constraints are then applied to the proposed vector of feature perturbations. Finally, the counterfactual search state is updated.
"""
function update!(counterfactual_explanation::CounterfactualExplanation) 

    # Generate peturbations:
    Δs′ = Generators.generate_perturbations(counterfactual_explanation.generator, counterfactual_explanation)
    Δs′ = apply_mutability(counterfactual_explanation, Δs′)         # mutability constraints
    s′ = counterfactual_explanation.s′ + Δs′                        # new proposed state
    apply_domain_constraints!(counterfactual_explanation)      # domain constraints
    
    # Updates:
    counterfactual_explanation.s′ = s′                                                  # update counterfactual
    _times_changed = reshape(decode_state(counterfactual_explanation, Δs′) .!= 0, size(counterfactual_explanation.search[:times_changed_features]))
    counterfactual_explanation.search[:times_changed_features] += _times_changed        # update number of times feature has been changed
    counterfactual_explanation.search[:mutability] = Generators.mutability_constraints(counterfactual_explanation.generator, counterfactual_explanation) 
    counterfactual_explanation.search[:iteration_count] += 1                            # update iteration counter   
    counterfactual_explanation.search[:path] = [counterfactual_explanation.search[:path]..., counterfactual_explanation.s′]
    counterfactual_explanation.search[:converged] = converged(counterfactual_explanation)
    counterfactual_explanation.search[:terminated] = terminated(counterfactual_explanation)

end

function Base.show(io::IO, z::CounterfactualExplanation)

    if  z.search[:iteration_count]>0
        if isnothing(z.params[:γ])
            p_path = target_probs_path(z)
            n_reached = findall([all(p .>= z.params[:γ]) for p in p_path])
            if length(n_reached) > 0 
                printstyled(io, "Threshold reached: $(all(threshold_reached(z)) ? "✅"  : "❌")", bold=true)
                print(" after $(first(n_reached)) steps.\n")
            end
            printstyled(io, "Convergence: $(converged(z) ? "✅"  : "❌")", bold=true)
            print(" after $(total_steps(z)) steps.\n")
        else
            printstyled(io, "Convergence: $(converged(z) ? "✅"  : "❌")", bold=true)
            print(" after $(total_steps(z)) steps.\n")
        end
    end

end

function Base.show(io::IO, z::Vector{CounterfactualExplanation})

    println(io,"")

end

