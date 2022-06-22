# ----- PyTorch Model ----- #
using PyCall
struct PyTorchModel <: AbstractDifferentiableModel
    model::Any
    type::Symbol
end

# Outer constructor method:
function PyTorchModel(model; type::Symbol=:classification_binary)
  PyTorchModel(model, type)
end

function logits(M::PyTorchModel, X::AbstractArray)
    py"""
    import torch
    from torch import nn
    """
    model = M.model
    if !isa(X, Matrix)
      X = reshape(X, length(X), 1)
    end
    ŷ = py"$model(torch.Tensor($X).T).detach().numpy()"
    ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]
    return ŷ'
end

function probs(M::PyTorchModel, X::AbstractArray)
    if M.type == :classification_binary
        output = σ.(logits(M, X))
    elseif M.type == :classification_multi
        output = softmax(logits(M, X))
    end
    return output
end