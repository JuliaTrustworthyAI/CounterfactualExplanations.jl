# -------- Schut et al (2020): 
struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    δ::AbstractFloat # perturbation size
    n::Int # maximum number of times any feature can be changed
end

# API streamlining:
using Parameters
@with_kw struct GreedyGeneratorParams
    δ::Union{AbstractFloat,Nothing}=nothing
    n::Union{Int,Nothing}=nothing
end

"""
    GreedyGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        δ::Union{AbstractFloat,Nothing}=nothing,
        n::Union{Int,Nothing}=nothing
    )

An outer constructor method that instantiates a greedy generator.

# Examples

```julia-repl
generator = GreedyGenerator()
```
"""
function GreedyGenerator(;loss::Union{Nothing,Symbol}=nothing,complexity::Function=norm,λ::AbstractFloat=0.0,kwargs...)

    # Load hyperparameters:
    params = GreedyGeneratorParams(;kwargs...)
    δ = params.δ
    n = params.n
    if all(isnothing.([δ, n])) 
        δ = 0.1
        n = 10
    elseif isnothing(δ) && !isnothing(n)
        δ = 1/n
    elseif !isnothing(δ) && isnothing(n)
        n = 1/δ
    end

    # Sanity checks:
    if λ != 0.0
        @warn "Choosing λ different from 0 has no effect on `GreedyGenerator`, since no penalty term is involved."
    end
    if complexity != norm
        @warn "Specifying `complexity` has no effect on `GreedyGenerator`, since no penalty term is involved."
    end

    generator = GreedyGenerator(loss,complexity,λ,δ,n)

    return generator
end

"""
    ∇(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)    

he default method to compute the gradient of the counterfactual search objective for a greedy generator. Since no complexity penalty is needed, this gradients just correponds to the partial derivative with respect to the loss function.

"""
∇(generator::GreedyGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State) = ∂ℓ(generator, M, counterfactual_state)

"""
    generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)

The default method to generate perturbations for a greedy generator. Only the most salient feature is perturbed.
"""
function generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State) 
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state) # gradient
    𝐠ₜ[counterfactual_state.params[:mutability] .== :none] .= 0
    function choose_most_salient(x)
        s = -((abs.(x).==maximum(abs.(x),dims=1)) .* generator.δ .* sign.(x))
        non_zero_elements = findall(vec(s).!=0)
        # If more than one equal, randomise:
        if length(non_zero_elements) > 1
            keep_ = rand(non_zero_elements)
            s_ = zeros(size(s))
            s_[keep_] = s[keep_]
            s = s_
        end
        return s
    end
    Δs′ = mapslices(x -> choose_most_salient(x), 𝐠ₜ, dims=1) # choose most salient feature
    return Δs′
end

"""
    mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)

The default method to return search state dependent mutability constraints for a greedy generator. Features that have been perturbed `n` times already can no longer be perturbed.
"""
function mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)
    mutability = counterfactual_state.params[:mutability]
    mutability[counterfactual_state.search[:times_changed_features] .>= generator.n] .= :none # constrains features that have already been exhausted
    return mutability
end 

"""
    conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)

If all features have been perturbed `n` times already, then the search terminates.
"""
function conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)
    status = all(map(times_changed -> all(times_changed.>=generator.n), counterfactual_state.search[:times_changed_features]))
    return status
end