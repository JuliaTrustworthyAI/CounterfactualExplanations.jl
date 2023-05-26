# Plan of implementing this:
# 1. Translate the following code into PythonCall from PyCall
# 2. Create a notebook where I can test this
# 3. Update a notebook where I test this with a PyTorch model

using PythonCall

struct PyTorchModel <: AbstractDifferentiableModel
    nn::Any
end

function logits(M::PyTorchModel, X::AbstractArray)
    torch = pyimport("torch")
    
    if !isa(X, Matrix)
      X = reshape(X, length(X), 1)
    end

    ŷ = M.nn(torch.Tensor(X).T).detach().numpy()
    
    ŷ = isa(ŷ, AbstractArray) ? ŷ : [ŷ]

    return ŷ'
end

probs(M::PyTorchModel, X::AbstractArray)= σ.(logits(M, X))