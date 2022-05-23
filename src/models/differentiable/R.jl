# ----- RTorch Model ----- #
using RCall

"""
    RTorchModel

Contructor for RTorch neural network.

"""
struct RTorchModel <: AbstractDifferentiableModel
    nn::Any
end

function logits(M::RTorchModel, X::AbstractArray)
  nn = M.nn
  ŷ = rcopy(R"as_array($nn(torch_tensor(t($X))))")
  ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]
  return ŷ'
end

probs(M::RTorchModel, X::AbstractArray)= σ.(logits(M, X))