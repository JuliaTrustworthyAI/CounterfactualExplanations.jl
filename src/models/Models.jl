# --------------- Base type for model:
module Models

using NNlib, LinearAlgebra

include("functions.jl")

export AbstractFittedModel, LogisticModel, BayesianLogisticModel, probs, logits

end