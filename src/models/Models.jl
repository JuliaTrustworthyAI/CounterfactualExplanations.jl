# --------------- Base type for model:
module Models

using Flux, LinearAlgebra

include("base.jl")

export AbstractFittedModel, AbstractDifferentiableModel, 
    FluxModel, LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

end