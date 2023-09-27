using CounterfactualExplanations.Generators

"""
    Generators.∂ℓ(generator::AbstractGradientBasedGenerator, M::PyTorchModel, ce::AbstractCounterfactualExplanation)

Method for computing the gradient of the loss function at the current counterfactual state for gradient-based generators operating on PyTorch models.
The gradients are calculated through PyTorch using PythonCall.jl.

# Arguments
- `generator::AbstractGradientBasedGenerator`: The generator object that is used to generate the counterfactual explanation.
- `M::Models.PyTorchModel`: The PyTorch model for which the counterfactual is generated.
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation object for which the gradient is calculated.

# Returns
- `grad::AbstractArray`: The gradient of the loss function at the current counterfactual state.

# Example
grad = ∂ℓ(generator, M, ce) # calculates the gradient of the loss function at the current counterfactual state.
"""
function Generators.∂ℓ(
    generator::AbstractGradientBasedGenerator,
    M::PyTorchModel,
    ce::AbstractCounterfactualExplanation,
)
    torch = PythonCall.pyimport("torch")
    np = PythonCall.pyimport("numpy")

    x = ce.x
    target = Float32.(ce.target_encoded)

    x = torch.tensor(np.array(reshape(x, 1, length(x))))
    x.requires_grad = true

    target = torch.tensor(np.array(reshape(target, 1, length(target))))
    target = target.squeeze()

    output = M.model(x).squeeze()

    obj_loss = torch.nn.BCEWithLogitsLoss()(output, target)
    obj_loss.backward()

    grad = PythonCall.pyconvert(Matrix, x.grad.t().detach().numpy())

    return grad
end
