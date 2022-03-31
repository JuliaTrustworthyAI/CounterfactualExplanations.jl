"""
    Recourse(x′::AbstractArray, y′::AbstractFloat, path::Matrix{AbstractFloat}, generator::Generators.AbstractGenerator, x::AbstractArray, y::AbstractFloat, M::Models.AbstractFittedModel, target::AbstractFloat)

Collects all variables relevant to the recourse outcome. 
"""
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

    return CounterfactualExplanation(x, target, nothing, x′, data, M, generator, params, nothing)

end

# Convenience methods:

# 0) Utils
output_dim(counterfactual_explanation::CounterfactualExplanation) = size(Models.probs(counterfactual_explanation.M, counterfactual_explanation.x))[1]

using Flux
function encode_target(counterfactual_explanation::CounterfactualExplanation) 
    out_dim = output_dim(counterfactual_explanation)
    target = counterfactual_explanation.target
    return out_dim > 1 ? Flux.onehot(target, 1:out_dim) : target
end

# 1) Factual values
factual(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.x
factual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.M, counterfactual_explanation.x)
p̅(counterfactual_explanation::CounterfactualExplanation) = factual_probability(counterfactual_explanation)
function factual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = p̅(counterfactual_explanation)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end
y(counterfactual_explanation::CounterfactualExplanation) = factual_label(counterfactual_explanation)

# 2) Counterfactual values:
function initialize!(counterfactual_explanation::CounterfactualExplanation) 

    # Encode target:
    counterfactual_explanation.target_encoded = encode_target(counterfactual_explanation)

    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 1,
        :times_changed_features => zeros(length(counterfactual_explanation.x)),
        :path => [counterfactual_explanation.x′],
        :terminated => threshold_reached(counterfactual_explanation),
        :converged => threshold_reached(counterfactual_explanation)
    )

    if counterfactual_explanation.search[:terminated]
        @info "Factual already in target class and probability exceeds threshold γ."
    end

end
outcome(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.x′
counterfactual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.M, counterfactual_explanation.x′)
p̲(counterfactual_explanation::CounterfactualExplanation) = counterfactual_probability(counterfactual_explanation)
function counterfactual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = p̲(counterfactual_explanation)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end
y′(counterfactual_explanation::CounterfactualExplanation) = counterfactual_label(counterfactual_explanation)

# 3) Search related methods:
terminated(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:terminated]
converged(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:converged]
total_steps(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:iteration_count]
path(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:path]

"""
    target_probs(p, target)

Selects the probabilities of the target class. In case of binary classification problem `p` reflects the probability that `y=1`. In that case `1-p` reflects the probability that `y=0`.

# Examples

```julia-repl
using CounterfactualExplanations
using CounterfactualExplanations.Models: LogisticModel, probs 
Random.seed!(1234)
N = 25
w = [1.0 1.0]# true coefficients
b = 0
x, y = toy_data_linear(N)
# Logit model:
M = LogisticModel(w, [b])
p = probs(M, x[rand(N)])
target_probs(p, 0)
target_probs(p, 1)
```

"""
function target_probs(counterfactual_explanation::CounterfactualExplanation, x::Union{AbstractArray, Nothing}=nothing)
    
    p = !isnothing(x) ? Models.probs(counterfactual_explanation.M, x) : p̲(counterfactual_explanation)
    target = counterfactual_explanation.target

    if length(p) == 1
        if target ∉ [0,1]
            throw(DomainError("For binary classification expecting target to be in {0,1}.")) 
        end
        # If target is binary (i.e. outcome 1D from sigmoid), compute p(y=0):
        p = vcat(1.0 .- p, p)
        # Choose first (target+1) row if target=0, second row (target+1) if target=1:  
        p_target = p[Int(target+1),:]
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
    apply_mutability(Δx′::AbstractArray, counterfactual.data::CounterfactualData, generator::AbstractGenerator, counterfactual.search::Dict)

Apply mutability constraints to `Δx′` based on vector of constraints `𝑭`.

# Examples 

𝑭 = [:both, :increase, :decrease, :none]
apply_mutability([-1,1,-1,1], 𝑭) # all but :none pass
apply_mutability([-1,-1,-1,1], 𝑭) # all but :increase and :none pass
apply_mutability([-1,1,1,1], 𝑭) # all but :decrease and :none pass
apply_mutability([-1,-1,1,1], 𝑭) # only :both passes

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

threshold_reached(counterfactual_explanation::CounterfactualExplanation) = target_probs(counterfactual_explanation)[1] >= counterfactual_explanation.params[:γ]
steps_exhausted(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:iteration_count] == counterfactual_explanation.params[:T]

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

function update!(counterfactual_explanation::CounterfactualExplanation) 

    counterfactual_state = get_counterfactual_state(counterfactual_explanation)

    # Generate peturbations:
    Δx′ = Generators.generate_perturbations(counterfactual_explanation.generator, counterfactual_state)
    Δx′ = apply_mutability(Δx′, counterfactual_explanation)
    Δx′ = reshape(Δx′, size(counterfactual_explanation.x′))
    x′ = counterfactual_explanation.x′ + Δx′
    x′ = DataPreprocessing.apply_domain_constraints(counterfactual_explanation.data, x′)
    counterfactual_explanation.x′ = x′ # update counterfactual
    # if !isnothing(feasible_range)
    #     clamp!(x′, feasible_range[1], feasible_range[2])
    # end
    
    # Updates:
    counterfactual_explanation.search[:path] = [counterfactual_explanation.search[:path]..., counterfactual_explanation.x′]
    counterfactual_explanation.search[:mutability] = Generators.mutability_constraints(counterfactual_explanation.generator, counterfactual_state) 
    counterfactual_explanation.search[:times_changed_features] += reshape(Δx′ .!= 0, size(counterfactual_explanation.search[:times_changed_features])) # update number of times feature has been changed
    counterfactual_explanation.search[:iteration_count] += 1 # update iteration counter   
    counterfactual_explanation.search[:converged] = threshold_reached(counterfactual_explanation)
    counterfactual_explanation.search[:terminated] = counterfactual_explanation.search[:converged] || steps_exhausted(counterfactual_explanation) || Generators.conditions_satisified(counterfactual_explanation.generator, counterfactual_state)
end

