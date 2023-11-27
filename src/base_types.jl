"Base type for counterfactual explanations."
abstract type AbstractCounterfactualExplanation end

"Base type for fitted models."
abstract type AbstractFittedModel end

"Treat `AbstractFittedModel` as scalar when broadcasting."
Base.broadcastable(model::AbstractFittedModel) = Ref(model)

"An abstract type that serves as the base type for counterfactual generators."
abstract type AbstractGenerator end

"Treat `AbstractGenerator` as scalar when broadcasting."
Base.broadcastable(gen::AbstractGenerator) = Ref(gen)

"An abstract type for parallelizers."
abstract type AbstractParallelizer end

"An abstract type that serves as the base type for convergence objects."
abstract type AbstractConvergenceType end
