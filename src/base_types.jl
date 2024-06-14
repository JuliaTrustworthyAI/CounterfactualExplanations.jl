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
