
################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

# ----- Julia models -----
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, counterfactual::Counterfactual)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.Models.AbstractDifferentiableJuliaModel, counterfactual::Counterfactual)
    gradient(() -> ℓ(generator, counterfactual), Flux.params(counterfactual.x′))[counterfactual.x′]
end

# ----- RTorch model -----
using RCall
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual::Counterfactual)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual::Counterfactual) 
    nn = M.nn
    x_cf = counterfactual.x′
    t = counterfactual.target_encoded
    R"""
    x <- torch_tensor($x_cf, requires_grad=TRUE)
    output <- $nn(x)
    obj_loss <- nnf_binary_cross_entropy_with_logits(output,$t)
    obj_loss$backward()
    """
    grad = rcopy(R"as_array(x$grad)")
    return grad
end

# ----- PyTorch model -----
using PyCall
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual::Counterfactual)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.PyTorchModel, counterfactual::Counterfactual) 
    py"""
    import torch
    from torch import nn
    """
    nn = M.nn
    x′ = counterfactual.x′
    t = counterfactual.target_encoded
    x = reshape(x′, 1, length(x′))
    py"""
    x = torch.Tensor($x)
    x.requires_grad = True
    t = torch.Tensor($[t]).squeeze()
    output = $nn(x).squeeze()
    obj_loss = nn.BCEWithLogitsLoss()(output,t)
    obj_loss.backward()
    """
    grad = vec(py"x.grad.detach().numpy()")
    return grad
end

"""
    ∂h(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂h(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual) = gradient(() -> h(generator, counterfactual), Flux.params(counterfactual.x′))[counterfactual.x′]

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual::Counterfactual)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators. It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual::Counterfactual) = ∂ℓ(generator, M, counterfactual) + generator.λ * ∂h(generator, counterfactual)

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual) 
    𝐠ₜ = ∇(generator, counterfactual.M, counterfactual) # gradient
    Δx′ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx′
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual)
    mutability = counterfactual.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

"""
    conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
"""
function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual::Counterfactual)
    𝐠ₜ = ∇(generator, counterfactual.M, counterfactual)
    status = all(abs.(𝐠ₜ) .< generator.τ) 
    return status
end

##################################################
# Specific Generators
##################################################

include("GenericGenerator/GenericGenerator.jl") # Wachter et al. (2017)
include("GreedyGenerator/GreedyGenerator.jl") # Schut et al. (2021)