# --------------- Base type for model:
abstract type FittedModel end

# -------- Linear model:
# This is an example of how to construct a FittedModel subtype:
using Flux
struct LogisticModel <: FittedModel
    w::AbstractArray
    b::AbstractArray
end

# What follows are the two required outer methods:
logits(ℳ::LogisticModel, X::AbstractArray) =  X * 𝓜.w .+ 𝓜.b
probs(ℳ::LogisticModel, X::AbstractArray) = Flux.σ.(logits(𝓜, X))