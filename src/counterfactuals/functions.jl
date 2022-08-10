mutable struct CounterfactualExplanation
    x::AbstractArray
    target::Number
    target_encoded::Union{Number, AbstractArray, Nothing}
    s′::AbstractArray
    f::Function
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractFittedModel
    generator::Generators.AbstractGenerator
    latent_space::Bool
    params::Dict
    search::Union{Dict,Nothing}
    num_counterfactuals::Int
    initialization::Symbol
end

using Statistics, Flux
"""
    CounterfactualExplanation(
        x::Union{AbstractArray,Int}, 
        target::Union{AbstractFloat,Int}, 
        data::CounterfactualData,  
        M::Models.AbstractFittedModel,
        generator::Generators.AbstractGenerator,
        γ::AbstractFloat, 
        T::Int
    )

Outer method to construct a `CounterfactualExplanation` structure.
"""
# Outer constructor method:
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

    @assert initialization ∈ [:identity, :add_perturbation]  

    # Factual:
    x = typeof(x) == Int ? select_factual(data, x) : x

    # Counterfactual state variable:
    size_ = Int.(vcat(ones(maximum([ndims(x),2])),num_counterfactuals))
    s′ = copy(x)  # start from factual
    s′ = repeat(x, outer=size_)
    # Initialization
    if initialization == :add_perturbation
        scale = std(data.X, dims=2) .* 0.1
        Δ = scale .* randn(size(scale,1))
        s′ = mapslices(s -> s .+ scale .* randn(size(scale,1)), s′, dims=(1,2))
    end
    f(s) = s # default mapping to feature space 

    # Parameters:
    params = Dict(
        :γ => generator.decision_threshold,
        :T => T,
        :mutability => repeat(DataPreprocessing.mutability_constraints(data), outer=size_),
        :initial_mutability => repeat(DataPreprocessing.mutability_constraints(data), outer=size_),
    )

    # Latent space:
    wants_latent_space = DataPreprocessing.has_pretrained_generative_model(data) || typeof(generator) <: Generators.AbstractLatentSpaceGenerator
    latent_space = isnothing(latent_space) ? wants_latent_space : latent_space

    # Instantiate: 
    counterfactual_explanation = CounterfactualExplanation(
        x, target, nothing, s′, f, 
        data, M, generator, latent_space, params, nothing, num_counterfactuals, initialization
    )

    # Encode target:
    counterfactual_explanation.target_encoded = encode_target(counterfactual_explanation)

    # Potential neighbours:
    ids = getindex.(findall(data.y.==counterfactual_explanation.target_encoded[:,:,1]),2)
    n_candidates = minimum([size(data.y,2),1000])
    candidates = select_factual(data,rand(ids,n_candidates))
    counterfactual_explanation.params[:potential_neighbours] = reduce(hcat, map(x -> x[1], collect(candidates)))

    # Check for redundancy:
    if threshold_reached(counterfactual_explanation)
        @info "Factual already in target class and probability exceeds threshold γ."
    end

    # Latent space:
    if counterfactual_explanation.latent_space
        @info "Searching in latent space using generative model."
        generative_model = DataPreprocessing.get_generative_model(counterfactual_explanation.data; generative_model_params...)
        # map counterfactual to latent space: s′=z′∼p(z|x)
        counterfactual_explanation.s′, _, _ = GenerativeModels.rand(generative_model.encoder, counterfactual_explanation.s′)

        # NOTE! This is not very clean, will be improved.
        if generative_model.params.nll==Flux.Losses.logitbinarycrossentropy
            counterfactual_explanation.f = function(s) Flux.σ.(generative_model.decoder(s)) end
        else
            counterfactual_explanation.f = function(s) generative_model.decoder(s) end
        end
    end

    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 0,
        :times_changed_features => zeros(size(counterfactual_explanation.f(counterfactual_explanation.s′))),
        :path => [counterfactual_explanation.s′],
        :terminated => threshold_reached(counterfactual_explanation),
        :converged => converged(counterfactual_explanation),
    )

    return counterfactual_explanation

end

# Convenience methods:

# 0) Utils
"""
    output_dim(counterfactual_explanation::CounterfactualExplanation)

A convenience method that computes the output dimension of the predictive model.
"""
output_dim(counterfactual_explanation::CounterfactualExplanation) = size(Models.probs(counterfactual_explanation.M, counterfactual_explanation.x))[1]

using Flux
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
counterfactual(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.f(counterfactual_explanation.s′)

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
    y = mapslices(p -> _to_label(p), p, dims=[1,2])
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
    counterfactual_state = get_counterfactual_state(counterfactual_explanation)
    if isnothing(counterfactual_explanation.generator.decision_threshold)
        threshold_reached(counterfactual_explanation) && Generators.conditions_satisified(counterfactual_explanation.generator, counterfactual_state)
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
        path = [counterfactual_explanation.f(z) for z ∈ path]
    end
    return path
end

"""
    counterfactual_probability_path(counterfactual_explanation::CounterfactualExplanation)

Returns the counterfactual probabilities for each step of the search.
"""
function counterfactual_probability_path(counterfactual_explanation::CounterfactualExplanation)
    M = counterfactual_explanation.M
    p = map(X -> mapslices(x -> probs(M, x), X, dims=[1,2]),path(counterfactual_explanation))
    return p
end

"""
    counterfactual_label_path(counterfactual_explanation::CounterfactualExplanation)

Returns the counterfactual labels for each step of the search.
"""
function counterfactual_label_path(counterfactual_explanation::CounterfactualExplanation)
    P = counterfactual_probability_path(counterfactual_explanation)
    ŷ = map(P -> mapslices(p -> _to_label(p), P, dims=[1,2]), P)
    return ŷ
end

"""
    target_probs_path(counterfactual_explanation::CounterfactualExplanation)

Returns the target probabilities for each step of the search.
"""
function target_probs_path(counterfactual_explanation::CounterfactualExplanation)
    X = path(counterfactual_explanation)
    P = map(X -> mapslices(x -> target_probs(counterfactual_explanation, x), X, dims=[1,2]), X)
    return P
end

using MLUtils
"""
    embed_path(counterfactual_explanation::CounterfactualExplanation)

Helper function that embeds path into two dimensions for plotting.
"""
function embed_path(counterfactual_explanation::CounterfactualExplanation)
    data_ = counterfactual_explanation.data
    path_ = stack(path(counterfactual_explanation),1)
    path_embedded = mapslices(X -> DataPreprocessing.embed(data_, X'), path_, dims=[1,2])
    path_embedded = unstack(path_embedded,dims=2)
    return path_embedded
end

"""
    apply_mutability(Δx′::AbstractArray, counterfactual_explanation::CounterfactualExplanation)

A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
"""
function apply_mutability(Δs′::AbstractArray, counterfactual_explanation::CounterfactualExplanation)

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
    threshold_reached(counterfactual_explanation::CounterfactualExplanation)

A convenience method that determines of the predefined threshold for the target class probability has been reached.
"""
function threshold_reached(counterfactual_explanation::CounterfactualExplanation)
    γ = isnothing(counterfactual_explanation.generator.decision_threshold) ? 0.5 : counterfactual_explanation.generator.decision_threshold
    all(target_probs(counterfactual_explanation) .>= γ)
end

"""
    steps_exhausted(counterfactual_explanation::CounterfactualExplanation) 

A convenience method that checks if the number of maximum iterations has been exhausted.
"""
steps_exhausted(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:iteration_count] == counterfactual_explanation.params[:T]

"""
    get_counterfactual_state(counterfactual_explanation::CounterfactualExplanation) 

A subroutine that is used to take a snapshot of the current counterfactual search state. This snapshot is passed to the counterfactual generator.
"""
function get_counterfactual_state(counterfactual_explanation::CounterfactualExplanation) 

    counterfactual_state = CounterfactualState.State(
        counterfactual_explanation.x,
        counterfactual_explanation.s′,
        counterfactual_explanation.f,
        counterfactual_label(counterfactual_explanation),
        counterfactual_explanation.target,
        counterfactual_explanation.target_encoded,
        counterfactual_explanation.params[:γ],
        threshold_reached(counterfactual_explanation),
        counterfactual_explanation.M,
        counterfactual_explanation.params,
        counterfactual_explanation.search
    )
    return counterfactual_state
end

"""
    update!(counterfactual_explanation::CounterfactualExplanation) 

An important subroutine that updates the counterfactual explanation. It takes a snapshot of the current counterfactual search state and passes it to the generator. Based on the current state the generator generates perturbations. Various constraints are then applied to the proposed vector of feature perturbations. Finally, the counterfactual search state is updated.
"""
function update!(counterfactual_explanation::CounterfactualExplanation) 

    counterfactual_state = get_counterfactual_state(counterfactual_explanation)

    # Generate peturbations:
    Δs′ = Generators.generate_perturbations(counterfactual_explanation.generator, counterfactual_state)
    
    if !counterfactual_explanation.latent_space
        Δs′ = apply_mutability(Δs′, counterfactual_explanation)
    else
        if !all(counterfactual_explanation.params[:mutability].==:both) && total_steps(counterfactual_explanation) == 0
            @warn "Mutability constraints not currently implemented for latent space search."
        end
    end

    s′ = counterfactual_explanation.s′ + Δs′
    if !counterfactual_explanation.latent_space
        s′ = DataPreprocessing.apply_domain_constraints(counterfactual_explanation.data, s′)
    else
        if !isnothing(counterfactual_explanation.data.domain) && total_steps(counterfactual_explanation) == 0
            @warn "Domain constraints not currently implemented for latent space search."
        end
    end
    counterfactual_explanation.s′ = s′ # update counterfactual
    
    # Updates:
    counterfactual_explanation.search[:times_changed_features] += reshape(counterfactual_explanation.f(Δs′) .!= 0, size(counterfactual_explanation.search[:times_changed_features])) # update number of times feature has been changed
    counterfactual_explanation.search[:mutability] = Generators.mutability_constraints(counterfactual_explanation.generator, counterfactual_state) 
    counterfactual_explanation.search[:iteration_count] += 1 # update iteration counter   
    counterfactual_explanation.search[:path] = [counterfactual_explanation.search[:path]..., counterfactual_explanation.s′]
    counterfactual_explanation.search[:converged] = converged(counterfactual_explanation)
    counterfactual_explanation.search[:terminated] = terminated(counterfactual_explanation)

end

function Base.show(io::IO, z::CounterfactualExplanation)

    if  z.search[:iteration_count]>0
        if isnothing(z.params[γ])
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

