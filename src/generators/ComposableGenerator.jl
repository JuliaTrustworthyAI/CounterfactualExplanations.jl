using Flux
using LinearAlgebra
using Parameters

"Class for Composable counterfactual generator following Wachter et al (2018)"
mutable struct ComposableGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}                       # loss function
    complexity::Union{Nothing,Function,Vector{Function}}        # penalties
    λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}       # strength of penalties
    decision_threshold::Union{Nothing,AbstractFloat}    # probability threshold
    opt::Flux.Optimise.AbstractOptimiser                # optimizer
    τ::AbstractFloat                                    # tolerance for convergence
end

# API streamlining:
@with_kw struct ComposableGeneratorParams
    opt::Flux.Optimise.AbstractOptimiser = Descent()
    τ::AbstractFloat = 1e-3
end

function ComposableGenerator()
    params = ComposableGeneratorParams()
    return ComposableGenerator(nothing, nothing, nothing, 0.5, params.opt, params.τ)
end

"""
    objective(generator, ex)

A macro that can be used to define the counterfactual search objective.
"""
macro objective(generator, ex)
    loss = getfield(CounterfactualExplanations.Objectives, ex.args[2])
    Λ = Vector{AbstractFloat}()
    costs = Vector{Function}()
    for i in 3:length(ex.args)
        ex_penalty = ex.args[i]
        λ = ex_penalty.args[2]
        push!(Λ, λ)
        cost = getfield(CounterfactualExplanations.Objectives, ex_penalty.args[3])
        push!(costs, cost)
    end
    ex_generator = quote
        $Main.generator.loss = $loss
        $Main.generator.complexity = $costs
        $Main.generator.λ = $Λ
        Main.generator
    end
    return ex_generator
end

"""
    threshold(generator, γ)

A simple macro that can be used to define the decision threshold `γ`.
"""
macro threshold(generator, γ)
    return :($Main.generator.decision_threshold = $γ; Main.generator)
end
