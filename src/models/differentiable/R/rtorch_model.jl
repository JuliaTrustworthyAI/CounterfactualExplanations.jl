struct RTorchModel <: AbstractRModel
    nn::Any
    likelihood::Symbol
end

function logits(model::RTorchModel, x::AbstractArray)
  if !isa(x, Matrix)
      x = reshape(x, length(x), 1)
  end

  model_nn = model.nn

  ŷ = rcopy(R"as_array($model_nn(torch_tensor(t($x))))")
  ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]

  return transpose(ŷ)
end

function probs(model::RTorchModel, x::AbstractArray) 
  if model.likelihood == :classification_binary
    return σ.(logits(model, x))
  elseif model.likelihood == :classification_multi
    return softmax(logits(model, x)) 
  end
end