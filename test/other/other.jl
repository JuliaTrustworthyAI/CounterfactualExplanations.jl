CounterfactualExplanations.reset!(flux_training_params)

@test typeof(Base.broadcastable(AbstractFittedModel)) <: Base.RefValue
@test typeof(Base.broadcastable(AbstractGenerator)) <: Base.RefValue
