mutable struct CounterfactualExplanation
    x::AbstractArray
    target::Number
    target_encoded::Union{Number, AbstractVector, Nothing}
    s′::AbstractArray
    f::Function
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractFittedModel
    generator::Generators.AbstractGenerator
    latent_space::Bool
    params::Dict
    search::Union{Dict,Nothing}
end

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
    x::Union{AbstractArray,Int}, 
    target::Union{AbstractFloat,Int}, 
    data::CounterfactualData,  
    M::Models.AbstractFittedModel,
    generator::Generators.AbstractGenerator,
    γ::AbstractFloat=0.75, 
    T::Int=100,
    latent_space::Bool=DataPreprocessing.has_pretrained_generative_model(data)
) 
    # Factual:
    x = typeof(x) == Int ? select_factual(data, x) : x

    # Counterfactual state variable:
    s′ = copy(x)  # start from factual
    f(s) = s # default mapping to feature space 

    # Parameters:
    params = Dict(
        :γ => γ,
        :T => T,
        :mutability => DataPreprocessing.mutability_constraints(data)
    )

    # Instantiate: 
    counterfactual_explantion = CounterfactualExplanation(
        x, target, nothing, s′, f, 
        data, M, generator, latent_space, params, nothing
    )

    # Encode target:
    counterfactual_explanation.target_encoded = encode_target(counterfactual_explanation)

    # Check for redundancy:
    if threshold_reached(counterfactual_explanation)
        @info "Factual already in target class and probability exceeds threshold γ."
    end

    # Latent space:
    if counterfactual_explantion.latent_space
        @info "Searching in latent space using generative model."
        generative_model = DataPreprocessing.get_generative_model(counterfactual_explanation.data)
        # map counterfactual to latent space: s′=z′∼p(z|x)
        counterfactual_explantion.s′ = rand(generative_model.encoder, counterfactual_explantion.x)
        f(s) = generative_model.decoder(s) # mapping from latent space
        counterfactual_explantion.f = f
    end

    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 1,
        :times_changed_features => zeros(length(counterfactual_explanation.x)),
        :path => [counterfactual_explanation.s′],
        :terminated => threshold_reached(counterfactual_explanation),
        :converged => threshold_reached(counterfactual_explanation),
    )

    return counterfactual_explantion

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
    return out_dim > 1 ? Flux.onehot(target, 1:out_dim) : target
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
counterfactual(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.f(counterfactual_explanation.z′)

"""
    counterfactual_probability(counterfactual_explanation::CounterfactualExplanation)

A convenience method to compute the class probabilities of the counterfactual value.
"""
counterfactual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.M, counterfactual(counterfactual_explanation))

"""
    counterfactual_label(counterfactual_explanation::CounterfactualExplanation) 

A convenience method to get the predicted label associated with the counterfactual value.
"""
function counterfactual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = counterfactual_probability(counterfactual_explanation)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end

# 3) Search related methods:
"""
    terminated(counterfactual_explanation::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has terminated.
"""
terminated(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:terminated]

"""
    converged(counterfactual_explanation::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has converged.
"""
converged(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:converged]

"""
    total_steps(counterfactual_explanation::CounterfactualExplanation)

A convenience method that returns the total number of steps of the counterfactual search.
"""
total_steps(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:iteration_count]

"""
    path(counterfactual_explanation::CounterfactualExplanation)

A convenience method that returns the entire counterfactual path.
"""
path(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:path]

"""
    target_probs(counterfactual_explanation::CounterfactualExplanation, x::Union{AbstractArray, Nothing}=nothing)

Returns the predicted probability of the target class for `x`. If `x` is `nothing`, the predicted probability corresponding to the counterfactual value is returned.
"""
function target_probs(counterfactual_explanation::CounterfactualExplanation, x::Union{AbstractArray, Nothing}=nothing)
    
    p = !isnothing(x) ? Models.probs(counterfactual_explanation.M, x) : counterfactual_probability(counterfactual_explanation)
    target = counterfactual_explanation.target

    if length(p) == 1
        h(x) = ifelse(x==-1,0,x)
        if target ∉ [0,1] && target ∉ [-1,1]
            throw(DomainError("For binary classification expecting target to be in {0,1} or {-1,1}.")) 
        end
        # If target is binary (i.e. outcome 1D from sigmoid), compute p(y=0):
        p = vcat(1.0 .- p, p)
        # Choose first (target+1) row if target=0, second row (target+1) if target=1:  
        p_target = p[Int(h(target)+1),:]
    else
        if target < 1 || target % 1 !=0
            throw(DomainError("For multi-class classification expecting `target` ∈ ℕ⁺, i.e. {1,2,3,...}.")) 
        end
        # If target is multi-class, choose corresponding row (e.g. target=2 -> row 2)
        p_target = p[Int(target),:]
    end
    return p_target
end

"""
    apply_mutability(Δx′::AbstractArray, counterfactual_explanation::CounterfactualExplanation)

A subroutine that applies mutability constraints to the proposed vector of feature perturbations.
"""
function apply_mutability(Δs′::AbstractArray, counterfactual_explanation::CounterfactualExplanation)

    mutability = counterfactual_explanation.params[:mutability]
    # Helper functions:
    both(x) = x
    increase(x) = ifelse(x<0,0,x)
    decrease(x) = ifelse(x>0,0,x)
    none(x) = 0
    cases = (both = both, increase = increase, decrease = decrease, none = none)

    # Apply:
    Δs′ = [getfield(cases, mutability[d])(counterfactual_explanation.f(Δs′[d])) for d in 1:length(Δs′)]

    return Δs′

end

"""
    threshold_reached(counterfactual_explanation::CounterfactualExplanation)

A convenience method that determines of the predefined threshold for the target class probability has been reached.
"""
threshold_reached(counterfactual_explanation::CounterfactualExplanation) = target_probs(counterfactual_explanation)[1] >= counterfactual_explanation.params[:γ]

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
        counterfactual_explanation.s′,
        counterfactual_explanation.f,
        counterfactual_explanation.target_encoded,
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
    Δs′ = apply_mutability(Δs′, counterfactual_explanation)
    Δs′ = reshape(Δs′, size(counterfactual_explanation.s′))
    s′ = counterfactual_explanation.s′ + Δs′
    # x′ = DataPreprocessing.apply_domain_constraints(counterfactual_explanation.data, x′)
    counterfactual_explanation.s′ = s′ # update counterfactual
    
    # Updates:
    counterfactual_explanation.search[:path] = [counterfactual_explanation.search[:path]..., counterfactual_explanation.s′]
    counterfactual_explanation.search[:mutability] = Generators.mutability_constraints(counterfactual_explanation.generator, counterfactual_state) 
    counterfactual_explanation.search[:times_changed_features] += reshape(Δs′ .!= 0, size(counterfactual_explanation.search[:times_changed_features])) # update number of times feature has been changed
    counterfactual_explanation.search[:iteration_count] += 1 # update iteration counter   
    counterfactual_explanation.search[:converged] = threshold_reached(counterfactual_explanation)
    counterfactual_explanation.search[:terminated] = counterfactual_explanation.search[:converged] || steps_exhausted(counterfactual_explanation) || Generators.conditions_satisified(counterfactual_explanation.generator, counterfactual_state)
end

function Base.show(io::IO, z::CounterfactualExplanation)

    printstyled(io, "Factual: ", bold=true)
    println(io, "x=$(z.x), y=$(factual_label(z)), p=$(factual_probability(z))")
    printstyled(io, "Target: ", bold=true)
    println(io, "target=$(z.target), γ=$(z.params[:γ])")

    if !isnothing(z.search)
        printstyled(io, "Counterfactual outcome: ", bold=true)
        println(io, "x′=$(z.x′), y′=$(counterfactual_label(z)), p′=$(counterfactual_probability(z))")
        printstyled(io, "Converged: $(converged(z) ? "✅"  : "❌") ", bold=true)
        println("after $(total_steps(z)) steps.")
    else
        @info "Search not yet initatiated."
    end

end