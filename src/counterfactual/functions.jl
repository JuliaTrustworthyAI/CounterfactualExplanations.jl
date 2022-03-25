"""
    Recourse(x̲::AbstractArray, y̲::AbstractFloat, path::Matrix{AbstractFloat}, generator::Generators.AbstractGenerator, x̅::AbstractArray, y̅::AbstractFloat, 𝑴::Models.AbstractFittedModel, target::AbstractFloat)

Collects all variables relevant to the recourse outcome. 
"""
mutable struct CounterfactualExplanation
    x̅::AbstractArray
    target::Number
    x̲::AbstractArray
    data::DataPreprocessing.CounterfactualData
    𝑴::Models.AbstractFittedModel
    generator::Generators.AbstractGenerator
    params::Dict
    search::Union{Dict,Nothing}
end

# Outer constructor method:
function CounterfactualExplanation(
    x̅::Union{AbstractArray,Int}, 
    target::Union{AbstractFloat,Int}, 
    data::CounterfactualData, 
    generator::AbstractGenerator, 
    𝑴::Models.AbstractFittedModel,
    γ::AbstractFloat, 
    T::Int
) 
    # Factual:
    x̅ = typeof(x̅) == Int ? select_factual(data, x̅) : x̅
    # Counterfactual:
    x̲ = copy(x̅)  # start from factual

    # Parameters:
    params = Dict(
        :γ => γ,
        :T => T,
        :mutability => DataPreprocessing.mutability_constraints(data)
    )

    return CounterfactualExplanation(x̅, target, x̲, data, generator, 𝑴, params, nothing)

end

# Convenience methods:

# 0) Utils
output_dim(counterfactual_explanation::CounterfactualExplanation) = size(Models.probs(counterfactual_explanation.𝑴, counterfactual_explanation.x̅))[1]
function target_encoded(counterfactual_explanation::CounterfactualExplanation) 
    out_dim = output_dim(counterfactual_explanation)
    target = counterfactual_explanation.target
    return out_dim > 1 ? Flux.onehot(target, 1:out_dim) : target
end

# 1) Factual values
factual(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.x̅
factual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.𝑴, counterfactual_explanation.x̅)
p̅(counterfactual_explanation::CounterfactualExplanation) = factual_probability(counterfactual_explanation)
function factual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = p̅(counterfactual_explanation)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end
y̅(counterfactual_explanation::CounterfactualExplanation) = factual_label(counterfactual_explanation)

# 2) Counterfactual values:
function initialize!(counterfactual_explanation::CounterfactualExplanation) 
    # Initialize search:
    counterfactual_explanation.search = Dict(
        :iteration_count => 1,
        :times_changed_features => zeros(length(counterfactual_explanation.x̅)),
        :path => [counterfactual_explanation.x̲],
        :terminated => threshold_reached(counterfactual_explanation),
        :converged => threshold_reached(counterfactual_explanation)
    )

    if counterfactual_explanation.search[:terminated]
        @info "Factual already in target class and probability exceeds threshold γ."
    end

end
outcome(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.x̲
counterfactual_probability(counterfactual_explanation::CounterfactualExplanation) = Models.probs(counterfactual_explanation.𝑴, counterfactual_explanation.x̲)
p̲(counterfactual_explanation::CounterfactualExplanation) = counterfactual_probability(counterfactual_explanation)
function counterfactual_label(counterfactual_explanation::CounterfactualExplanation) 
    p = p̲(counterfactual_explanation)
    out_dim = size(p)[1]
    y = out_dim == 1 ? round(p[1]) : Flux.onecold(p,1:out_dim)
    return y
end
y̲(counterfactual_explanation::CounterfactualExplanation) = counterfactual_label(counterfactual_explanation)

# 3) Search related methods:
terminated(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:terminated]
converged(counterfactual_explanation::CounterfactualExplanation) = counterfactual_explanation.search[:converged]

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
𝑴 = LogisticModel(w, [b])
p = probs(𝑴, x[rand(N)])
target_probs(p, 0)
target_probs(p, 1)
```

"""
function target_probs(counterfactual_explanation::CounterfactualExplanation)
    p = p̲(counterfactual_explanation) # counterfactual probabilities
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
    apply_mutability(Δx̲::AbstractArray, counterfactual.data::CounterfactualData, generator::AbstractGenerator, counterfactual.search::Dict)

Apply mutability constraints to `Δx̲` based on vector of constraints `𝑭`.

# Examples 

𝑭 = [:both, :increase, :decrease, :none]
apply_mutability([-1,1,-1,1], 𝑭) # all but :none pass
apply_mutability([-1,-1,-1,1], 𝑭) # all but :increase and :none pass
apply_mutability([-1,1,1,1], 𝑭) # all but :decrease and :none pass
apply_mutability([-1,-1,1,1], 𝑭) # only :both passes

"""
function apply_mutability(Δx̲::AbstractArray, counterfactual_explanation::CounterfactualExplanation)

    mutability = counterfactual_explanation.params[:mutability]
    # Helper functions:
    both(x) = x
    increase(x) = ifelse(x<0,0,x)
    decrease(x) = ifelse(x>0,0,x)
    none(x) = 0
    cases = (both = both, increase = increase, decrease = decrease, none = none)

    # Apply:
    Δx̲ = [getfield(cases, mutability[d])(Δx̲[d]) for d in 1:length(Δx̲)]

    return Δx̲

end

threshold_reached(counterfactual_explanation::CounterfactualExplanation) = target_probs(counterfactual_explanation)[1] >= γ

function get_counterfactual_state(counterfactual_explanation::CounterfactualExplanation) 
    counterfactual_state = Generators.CounterfactualState(
        counterfactual_explanation.x̅,
        counterfactual_explanation.target,
        counterfactual_explanation.x̲,
        counterfactual_explanation.𝑴,
        counterfactual_explanation.params,
        counterfactual_explanation.search
    )
    return counterfactual_state
end

function update!(counterfactual_explanation::CounterfactualExplanation) 

    counterfactual_state = get_counterfactual_state(counterfactual_explanation)

    # Generate peturbations:
    Δx̲ = Generators.generate_perturbations(generator, counterfactual_state)
    Δx̲ = apply_mutability(Δx̲, counterfactual_explanation)
    Δx̲ = reshape(Δx̲, size(counterfactual_explanation.x̲))
    counterfactual_explanation.x̲ += Δx̲ # update counterfactual
    if !isnothing(feasible_range)
        clamp!(x̲, feasible_range[1], feasible_range[2])
    end
    
    # Updates:
    counterfactual_explanation.search[:path] = [counterfactual_explanation.search[:path]..., x̲]
    counterfactual_explanation.search[:mutability] = Generators.mutability_constraints(generator, counterfactual_state) 
    counterfactual_explanation.search[:times_changed_features] += reshape(Δx̲ .!= 0, size(counterfactual_explanation.search[:times_changed_features])) # update number of times feature has been changed
    counterfactual_explanation.search[:iteration_count] += 1 # update iteration counter   
    counterfactual_explanation.search[:converged] = threshold_reached(counterfactual_explanation)
    counterfactual_explanation.search[:terminated] = counterfactual_explanation.search[:converged] || t == T || Generators.conditions_satisified(generator, counterfactual_state)
end
