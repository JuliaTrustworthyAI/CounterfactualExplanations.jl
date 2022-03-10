# --------------- Base type for model:
module Models

using NNlib, LinearAlgebra

include("functions.jl")

export LogisticModel, BayesianLogisticModel, probs, logits

end