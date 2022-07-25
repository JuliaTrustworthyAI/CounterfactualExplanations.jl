# --------------- Base type for model:
module Models

using Flux, LinearAlgebra
using ..Interoperability, ..DataPreprocessing

include("base.jl")

export AbstractFittedModel, AbstractDifferentiableModel, 
    FluxModel, LaplaceReduxModel, LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

include("plotting.jl")

end