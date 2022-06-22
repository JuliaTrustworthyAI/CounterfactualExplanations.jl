# ----- RTorch Model ----- #
using RCall

"""
    RTorchModel

Contructor for RTorch neural network.

"""
struct RTorchModel <: AbstractDifferentiableModel
    model::Any
    type::Symbol
end

# Outer constructor method:
function RTorchModel(model; type::Symbol=:classification_binary)
  RTorchModel(model, type)
end

function logits(M::RTorchModel, X::AbstractArray)
  model = M.model
  ŷ = rcopy(R"as_array($model(torch_tensor(t($X))))")
  ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]
  return ŷ'
end

function probs(M::RTorchModel, X::AbstractArray)
  if M.type == :classification_binary
      output = σ.(logits(M, X))
  elseif M.type == :classification_multi
      output = softmax(logits(M, X))
  end
  return output
end