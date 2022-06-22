# ----- RTorch Model ----- #
using RCall

"""
    RTorchModel

Contructor for RTorch neural network.

"""
struct RTorchModel <: AbstractDifferentiableModel
    model::Any
    likelihood::Symbol
end

# Outer constructor method:
function RTorchModel(model; likelihood::Symbol=:classification_binary)
  RTorchModel(model, likelihood)
end

function logits(M::RTorchModel, X::AbstractArray)
  model = M.model
  if size(X)[1] == 1
      X = X'
  end
  if !isa(X, Matrix)
    X = reshape(X, length(X), 1)
  end
  ŷ = rcopy(R"as_array($model(torch_tensor(t($X))))")
  ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]
  return ŷ'
end

function probs(M::RTorchModel, X::AbstractArray)
  if M.likelihood == :classification_binary
      output = σ.(logits(M, X))
  elseif M.likelihood == :classification_multi
      output = softmax(logits(M, X))
  end
  return output
end