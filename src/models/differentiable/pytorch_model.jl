using PythonCall

"""
PyTorchModel <: AbstractDifferentiablePythonModel

Constructor for models trained in `PyTorch`. 
"""
struct PyTorchModel <: AbstractDifferentiablePythonModel
    neural_network::Any
    likelihood::Symbol
end

"""
    function logits(model::PyTorchModel, x::AbstractArray)

Calculates the logit scores output by the model `model` for the input data `X`.

# Arguments
- `model::PyTorchModel`: The model selected by the user. Must be a model defined using PyTorch.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The logit scores for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data points in `X`
"""
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

"""
    function probs(model::PyTorchModel, x::AbstractArray)

Calculates the output probabilities of the model `model` for the input data `X`.

# Arguments
- `model::PyTorchModel`: The model selected by the user. Must be a model defined using PyTorch.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The probabilities for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the probabilities for each output class for the data points in `X`
"""
function probs(model::PyTorchModel, x::AbstractArray)
    if model.likelihood == :classification_binary
        return σ.(logits(model, x))
    elseif model.likelihood == :classification_multi
        return softmax(logits(model, x))
    end
end
