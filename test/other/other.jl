CounterfactualExplanations.reset!(flux_training_params)

@test typeof(Base.broadcastable(AbstractModel)) <: Base.RefValue
@test typeof(Base.broadcastable(AbstractGenerator)) <: Base.RefValue
