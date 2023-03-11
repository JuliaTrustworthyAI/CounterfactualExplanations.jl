using Parameters
using LinearAlgebra
using SliceMap

"Class for Greedy counterfactual generator following Schut et al (2020)."
mutable struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function} # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    decision_threshold::Union{Nothing,AbstractFloat} # probability threshold
    ϵ::AbstractFloat # learning rate
    τ::AbstractFloat # tolerance for convergence
    n::Int # maximum number of times any feature can be changed
    passes::Int # number of full passes (`n` times) through all features
end

# API streamlining:
@with_kw struct GreedyGeneratorParams
    ϵ::Union{AbstractFloat,Nothing} = nothing
    τ::AbstractFloat = 1e-3
    n::Union{Int,Nothing} = nothing
end

"""
    GreedyGenerator(;
        loss::Union{Nothing,Function} = nothing,
        complexity::Function = LinearAlgebra.norm,
        λ::AbstractFloat = 0.0,
        decision_threshold = 0.5,
        opt::Union{Nothing,Flux.Optimise.AbstractOptimiser} = nothing, # learning rate
        kwargs...,
    )

An outer constructor method that instantiates a greedy generator.

# Examples

```julia-repl
generator = GreedyGenerator()
```
"""
function GreedyGenerator(;
    loss::Union{Nothing,Function}=nothing,
    complexity::Function=LinearAlgebra.norm,
    λ::AbstractFloat=0.0,
    decision_threshold=0.5,
    opt::Union{Nothing,Flux.Optimise.AbstractOptimiser}=nothing, # learning rate
    kwargs...
)

    if !isnothing(opt)
        @warn "The `GreedyGenerator` does not not work with a `Flux` optimiser. Argument `opt` will be ignored."
        opt = nothing
    end

    # Load hyperparameters:
    params = GreedyGeneratorParams(; kwargs...)
    ϵ = params.ϵ
    n = params.n
    if all(isnothing.([ϵ, n]))
        ϵ = 0.1
        n = 10
    elseif isnothing(ϵ) && !isnothing(n)
        ϵ = 1 / n
    elseif !isnothing(ϵ) && isnothing(n)
        n = 1 / ϵ
    end

    # Sanity checks:
    if λ != 0.0
        @warn "Choosing λ different from 0 has no effect on `GreedyGenerator`, since no penalty term is involved."
    end
    if complexity != LinearAlgebra.norm
        @warn "Specifying `complexity` has no effect on `GreedyGenerator`, since no penalty term is involved."
    end

    generator = GreedyGenerator(loss, complexity, λ, decision_threshold, ϵ, params.τ, n, 0)

    return generator
end

"""
    ∇(generator::GreedyGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)    

he default method to compute the gradient of the counterfactual search objective for a greedy generator. Since no complexity penalty is needed, this gradients just correponds to the partial derivative with respect to the loss function.

"""
∇(
    generator::GreedyGenerator,
    M::Models.AbstractDifferentiableModel,
    counterfactual_explanation::AbstractCounterfactualExplanation,
) = ∂ℓ(generator, M, counterfactual_explanation)

"""
    generate_perturbations(generator::GreedyGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to generate perturbations for a greedy generator. Only the most salient feature is perturbed.
"""
function generate_perturbations(
    generator::GreedyGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)
    𝐠ₜ = ∇(generator, counterfactual_explanation.M, counterfactual_explanation) # gradient
    𝐠ₜ[counterfactual_explanation.params[:mutability].==:none] .= 0
    function choose_most_salient(x)
        s = -((abs.(x) .== maximum(abs.(x), dims=1)) .* generator.ϵ .* sign.(x))
        non_zero_elements = findall(vec(s) .!= 0)
        # If more than one equal, randomise:
        if length(non_zero_elements) > 1
            keep_ = rand(non_zero_elements)
            s_ = zeros(size(s))
            s_[keep_] = s[keep_]
            s = s_
        end
        return s
    end
    Δs′ = SliceMap.slicemap(x -> choose_most_salient(x), 𝐠ₜ, dims = 1) # choose most salient feature
    Δs′ = convert.(eltype(counterfactual_explanation.x), Δs′)
    return Δs′
end

"""
    mutability_constraints(generator::GreedyGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to return search state dependent mutability constraints for a greedy generator. Features that have been perturbed `n` times already can no longer be perturbed.
"""
function mutability_constraints(
    generator::GreedyGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)
    mutability = counterfactual_explanation.params[:mutability]
    if all(counterfactual_explanation.search[:times_changed_features] .>= generator.n)
        generator.passes += 1
        generator.n += generator.n / generator.passes
        @info "Steps exhausted for all mutable features. Increasing number of allowed steps to $(generator.n). Restoring initial mutability."
        counterfactual_explanation.params[:mutability] .=
            counterfactual_explanation.params[:initial_mutability]
    end
    mutability[counterfactual_explanation.search[:times_changed_features].>=generator.n] .=
        :none # constrains features that have already been exhausted
    return mutability
end
