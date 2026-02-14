"Base type for counterfactual explanations."
abstract type AbstractCounterfactualExplanation end

"Base type for models."
abstract type AbstractModel end

"Treat `AbstractModel` as scalar when broadcasting."
Base.broadcastable(model::AbstractModel) = Ref(model)

"An abstract type that serves as the base type for counterfactual generators."
abstract type AbstractGenerator end

"Treat `AbstractGenerator` as scalar when broadcasting."
Base.broadcastable(gen::AbstractGenerator) = Ref(gen)

"An abstract type that serves as the base type for convergence objects."
abstract type AbstractConvergence end

"An abstract type that serves as the base type for measures. Objects of type `AbstractMeasure` need to be callable."
abstract type AbstractMeasure <: Function end

measure_name(m::Function) = Symbol(m)

"An abstract type for penalty functions."
abstract type AbstractPenalty <: AbstractMeasure end

"Treat `AbstractPenalty` as scalar when broadcasting."
Base.broadcastable(pen::AbstractPenalty) = Ref(pen)

const PenaltyOrFun = Union{Function,AbstractPenalty}

flatten_penalty(pen::PenaltyOrFun) = [pen]
flatten_penalty(pen::Vector{<:PenaltyOrFun}) = pen
flatten_penalty(pen::Vector{<:Tuple}) = [p[1] for p in pen]
