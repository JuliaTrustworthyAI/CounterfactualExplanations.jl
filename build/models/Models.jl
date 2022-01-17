# --------------- Base type for model:
module Models

using Flux

abstract type FittedModel end


# -------- Linear model:
# This is an example of how to construct a FittedModel subtype:
struct LogisticModel <: FittedModel
    w::AbstractArray
    b::AbstractArray
end

# What follows are the two required outer methods:
logits(𝑴::LogisticModel, X::AbstractArray) = X * 𝑴.w .+ 𝑴.b
probs(𝑴::LogisticModel, X::AbstractArray) = Flux.σ.(logits(𝑴, X))

# -------- Bayesian model:
struct BayesianLogisticModel <: FittedModel
end
end