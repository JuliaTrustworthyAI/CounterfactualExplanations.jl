"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractDifferentiableModel,
    ce::AbstractCounterfactualExplanation,
)
    gs = 0
    gs = Flux.gradient(() -> ℓ(generator, ce), Flux.params(ce.s′))[ce.s′]
    return gs
end

"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.PyTorchModel, ce::AbstractCounterfactualExplanation)

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
function ∂ℓ(
    generator::AbstractGradientBasedGenerator,
    M::Models.PyTorchModel,
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



"""
    ∂h(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators.
It assumes that `Zygote.jl` has gradient access.
"""
function ∂h(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    return Flux.gradient(() -> h(generator, ce), Flux.params(ce.s′))[ce.s′]
end

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators.
It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
If the counterfactual is being generated using Probe, the hinge loss is added to the gradient.
"""
function ∇(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractDifferentiableModel,
    ce::AbstractCounterfactualExplanation,
)
    ℓ = 0
    if (ce.convergence[:converge_when] == :invalidation_rate)
        ℓ = hinge_loss(ce)
    end
    return ∂ℓ(generator, M, ce) + ∂h(generator, ce) .+ ℓ
end
