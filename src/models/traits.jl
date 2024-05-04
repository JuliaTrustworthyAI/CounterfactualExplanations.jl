"A base type for model differentiability."
abstract type Differentiability end

"Dispatches on the type of model for the differentiability trait."
Differentiability(M::Model) = Differentiability(M.type)

"By default, models are assumed not to be differentiable."
struct NonDifferentiable <: Differentiability end

Differentiability(::AbstractModelType) = NonDifferentiable()

"Struct for models that are differentiable."
struct IsDifferentiable <: Differentiability end