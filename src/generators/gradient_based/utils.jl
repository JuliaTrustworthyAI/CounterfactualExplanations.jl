using Flux
using Statistics
using PythonCall

"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, ce::AbstractCounterfactualExplanation)

Method for computing the gradient of the loss function at the current counterfactual state for gradient-based generators operating on Flux models.
It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractFluxModel,
    ce::AbstractCounterfactualExplanation,
)
    gs = gradient(() -> ℓ(generator, ce), Flux.params(ce.s′))[ce.s′]
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

    output = M.neural_network(x).squeeze()

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
    return gradient(() -> h(generator, ce), Flux.params(ce.s′))[ce.s′]
end

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, ce::AbstractCounterfactualExplanation)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators.
It simply computes the weighted sum over partial derivates.
It assumes that `Zygote.jl` has gradient access.
"""
function ∇(
    generator::AbstractGradientBasedGenerator,
    M::Models.AbstractDifferentiableModel,
    ce::AbstractCounterfactualExplanation,
)
    return ∂ℓ(generator, M, ce) + ∂h(generator, ce)
end

"""
    propose_state(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

Proposes new state based on backpropagation.
"""
function propose_state(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    grads = ∇(generator, ce.M, ce) # gradient
    new_s′ = deepcopy(ce.s′)
    Flux.Optimise.update!(generator.opt, new_s′, grads)
    return new_s′
end

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    s′ = deepcopy(ce.s′)
    new_s′ = propose_state(generator, ce)
    Δs′ = new_s′ - s′                                           # gradient step
    Δs′ = _replace_nans(Δs′)
    Δs′ *= ce.num_counterfactuals       # rescale to account for number of counterfactuals
    Δs′ = convert.(eltype(ce.x), Δs′)

    return Δs′
end

"""
    _replace_nans(Δs′::AbstractArray, old_new::Pair=(NaN => 0))

Helper function to deal with exploding gradients. This is only a temporary fix and will be improved.
"""
function _replace_nans(Δs′::AbstractArray, old_new::Pair=(NaN => 0))
    return replace(Δs′, old_new)
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to return mutability constraints that are dependent on the current counterfactual search state.
For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    mutability = ce.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end

"""
    conditions_satisfied(generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
By default, gradient-based search is considered to have converged as soon as the proposed feature changes for all features are smaller than one percent of its standard deviation.
"""
function conditions_satisfied(
    generator::AbstractGradientBasedGenerator, ce::AbstractCounterfactualExplanation
)
    Δs′ = ∇(generator, ce.M, ce)
    Δs′ = CounterfactualExplanations.apply_mutability(ce, Δs′)
    τ = ce.convergence[:gradient_tol]
    satisfied = map(x -> all(abs.(x) .< τ), eachslice(Δs′; dims=ndims(Δs′)))
    success_rate = sum(satisfied) / ce.num_counterfactuals
    status = success_rate > ce.convergence[:min_success_rate]
    return status
end
