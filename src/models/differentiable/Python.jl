# ----- PyTorch Model ----- #
using PyCall
struct PyTorchModel <: AbstractDifferentiableModel
    nn::Any
end

function logits(M::PyTorchModel, X::AbstractArray)
    py"""
    import torch
    from torch import nn
    """
    nn = M.nn
    if !isa(X, Matrix)
      X = reshape(X, length(X), 1)
    end
    ŷ = py"$nn(torch.Tensor($X).T).detach().numpy()"
    ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]
    return ŷ'
end

probs(M::PyTorchModel, X::AbstractArray)= σ.(logits(M, X))