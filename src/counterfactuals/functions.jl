mutable struct CounterfactualExplanation
    x::AbstractArray
    target::Number
    target_encoded::Union{Number, AbstractVector, Nothing}
    x′::AbstractArray
    data::DataPreprocessing.CounterfactualData
    M::Models.AbstractFittedModel
    generator::Generators.AbstractGenerator
    params::Dict
    search::Union{Dict,Nothing}
end

using .Counterfactuals
function Base.show(io::IO, z::CounterfactualExplanation)

    printstyled(io, "Factual: ", bold=true)
    println(io, "x=$(z.x), y=$(Counterfactuals.factual_label(z)), p=$(Counterfactuals.factual_probability(z))")
    printstyled(io, "Target: ", bold=true)
    println(io, "target=$(z.target), γ=$(z.params[:γ])")

    if !isnothing(z.search)
        printstyled(io, "Counterfactual outcome: ", bold=true)
        println(io, "x′=$(z.x′), y′=$(Counterfactuals.counterfactual_label(z)), p′=$(Counterfactuals.counterfactual_probability(z))")
        printstyled(io, "Converged: $(Counterfactuals.converged(z) ? "✅"  : "❌") ", bold=true)
        println("after $(Counterfactuals.total_steps(z)) steps.")
    else
        @info "Search not yet initatiated."
    end

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
    x::Union{AbstractArray,Int}, 
    target::Union{AbstractFloat,Int}, 
    data::CounterfactualData,  
    M::Models.AbstractFittedModel,
    generator::Generators.AbstractGenerator,
    γ::AbstractFloat, 
    T::Int
) 
    # Factual:
    x = typeof(x) == Int ? select_factual(data, x) : x
    # Counterfactual:
    x′ = copy(x)  # start from factual

    # Parameters:
    params = Dict(
        :γ => γ,
        :T => T,
        :mutability => DataPreprocessing.mutability_constraints(data)
    )

    # Instantiate: 
    counterfactual_explantion = CounterfactualExplanation(x, target, nothing, x′, data, M, generator, params, nothing)

    initialize!(counterfactual_explantion) 

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
    initialize!(counterfactual_explanation::CounterfactualExplanation, generator::AbstractGenerator) 

Default subroutine that intializes the counterfactual search.
"""
function initialize!(counterfactual_explanation::CounterfactualExplanation, generator::AbstractGenerator) 

    # Encode target:
    counterfactual_explanation.target_encoded = encode_target(counterfactual_explanation)

    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 1,
        :times_changed_features => zeros(length(counterfactual_explanation.x)),
        :path => [counterfactual_explanation.x′],
        :terminated => threshold_reached(counterfactual_explanation),
        :converged => threshold_reached(counterfactual_explanation),
        :other => nothing
    )

    if counterfactual_explanation.search[:terminated]
        @info "Factual already in target class and probability exceeds threshold γ."
    end

end

"""
    initialize!(counterfactual_explanation::CounterfactualExplanation, generator::LatentSpaceGenerator) 

Default subroutine that intializes the counterfactual search.
"""
function initialize!(counterfactual_explanation::CounterfactualExplanation, generator::LatentSpaceGenerator) 

    # Encode target:
    counterfactual_explanation.target_encoded = encode_target(counterfactual_explanation)

    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 1,
        :times_changed_features => zeros(length(counterfactual_explanation.x)),
        :path => [counterfactual_explanation.x′],
        :terminated => threshold_reached(counterfactual_explanation),
        :converged => threshold_reached(counterfactual_explanation),
        :other => nothing
    )

    if counterfactual_explanation.search[:terminated]
        @info "Factual already in target class and probability exceeds threshold γ."
    end

    # Generative model:
    if generator.train_gm
        @info "Training generative model."
        GenerativeModels.train(generator.generative_model)
    else 
        @info "Skipping training of generative model."
    end

end

# Just a wrapper for the outer API (following https://discourse.julialang.org/t/best-way-to-dispatch-on-type-field/4024/2)
initialize!(counterfactual_explanation::CounterfactualExplanation) = initialize!(counterfactual_explanation, counterfactual_explanation.generator)

"""
    counterfactual(counterfactual_explanation::CounterfactualExplanation)

A convenience method to get the counterfactual value.
"""
counterfactual(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.x′

"""
    counterfactual_probability(counterfactual_explanation::CounterfactualExplanation)

A convenience method to compute the class probabilities of the counterfactual value.
"""
counterfactual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.M, counterfactual_explanation.x′)

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
function apply_mutability(Δx′::AbstractArray, counterfactual_explanation::CounterfactualExplanation)

    mutability = counterfactual_explanation.params[:mutability]
    # Helper functions:
    both(x) = x
    increase(x) = ifelse(x<0,0,x)
    decrease(x) = ifelse(x>0,0,x)
    none(x) = 0
    cases = (both = both, increase = increase, decrease = decrease, none = none)

    # Apply:
    Δx′ = [getfield(cases, mutability[d])(Δx′[d]) for d in 1:length(Δx′)]

    return Δx′

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
    counterfactual_state = Generators.CounterfactualState(
        counterfactual_explanation.x,
        counterfactual_explanation.target_encoded,
        counterfactual_explanation.x′,
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
    Δx′ = Generators.generate_perturbations(counterfactual_explanation.generator, counterfactual_state)
    Δx′ = apply_mutability(Δx′, counterfactual_explanation)
    Δx′ = reshape(Δx′, size(counterfactual_explanation.x′))
    x′ = counterfactual_explanation.x′ + Δx′
    x′ = DataPreprocessing.apply_domain_constraints(counterfactual_explanation.data, x′)
    counterfactual_explanation.x′ = x′ # update counterfactual
    
    # Updates:
    counterfactual_explanation.search[:path] = [counterfactual_explanation.search[:path]..., counterfactual_explanation.x′]
    counterfactual_explanation.search[:mutability] = Generators.mutability_constraints(counterfactual_explanation.generator, counterfactual_state) 
    counterfactual_explanation.search[:times_changed_features] += reshape(Δx′ .!= 0, size(counterfactual_explanation.search[:times_changed_features])) # update number of times feature has been changed
    counterfactual_explanation.search[:iteration_count] += 1 # update iteration counter   
    counterfactual_explanation.search[:converged] = threshold_reached(counterfactual_explanation)
    counterfactual_explanation.search[:terminated] = counterfactual_explanation.search[:converged] || steps_exhausted(counterfactual_explanation) || Generators.conditions_satisified(counterfactual_explanation.generator, counterfactual_state)
end

