using PythonCall

struct PyTorchModel <: AbstractDifferentiablePythonModel
    neural_network::Any
    likelihood::Symbol
end

function logits(model::PyTorchModel, x::AbstractArray)
    torch = PythonCall.pyimport("torch")
    np = PythonCall.pyimport("numpy")

    if !isa(x, Matrix)
        x = reshape(x, length(x), 1)
    end

    ŷ_python = model.neural_network(torch.tensor(np.array(x)).T).detach().numpy()
    ŷ = PythonCall.pyconvert(Matrix, ŷ_python)

    return transpose(ŷ)
end

function probs(model::PyTorchModel, x::AbstractArray)
    if model.likelihood == :classification_binary
        return σ.(logits(model, x))
    elseif model.likelihood == :classification_multi
        return softmax(logits(model, x))
    end
end
