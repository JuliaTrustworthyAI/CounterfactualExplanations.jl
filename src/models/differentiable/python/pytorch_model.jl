"""
PyTorchModel <: AbstractPythonModel

Constructor for models trained in `PyTorch`. 
"""
struct PyTorchModel <: AbstractPythonModel
    model::Any
    likelihood::Symbol
    function PyTorchModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi]`"
                ),
            )
        end
    end
end

"""
    function logits(M::PyTorchModel, x::AbstractArray)

Calculates the logit scores output by the model `M` for the input data `X`.

# Arguments
- `M::PyTorchModel`: The model selected by the user. Must be a model defined using PyTorch.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The logit scores for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the logit scores for each output class for the data points in `X`
"""
function logits(M::PyTorchModel, x::AbstractArray)
    torch = PythonCall.pyimport("torch")
    np = PythonCall.pyimport("numpy")

    if !isa(x, Matrix)
        x = reshape(x, length(x), 1)
    end

    ŷ_python = M.model(torch.tensor(np.array(x)).T).detach().numpy()
    ŷ = PythonCall.pyconvert(Matrix, ŷ_python)

    return transpose(ŷ)
end

"""
    function probs(M::PyTorchModel, x::AbstractArray)

Calculates the output probabilities of the model `M` for the input data `X`.

# Arguments
- `M::PyTorchModel`: The model selected by the user. Must be a model defined using PyTorch.
- `X::AbstractArray`: The feature vector for which the logit scores are calculated.

# Returns
- `logits::AbstractArray`: The probabilities for each output class for the data points in `X`.

# Example
logits = Models.logits(M, x) # calculates the probabilities for each output class for the data points in `X`
"""
function probs(M::PyTorchModel, x::AbstractArray)
    if M.likelihood == :classification_binary
        return σ.(logits(M, x))
    elseif M.likelihood == :classification_multi
        return softmax(logits(M, x))
    end
end
