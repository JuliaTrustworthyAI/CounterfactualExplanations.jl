
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
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Union{Models.LogisticModel, Models.BayesianLogisticModel}, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State)
    gs = gradient(() -> ℓ(generator, counterfactual_state), Flux.params(counterfactual_state.s′))[counterfactual_state.s′]
    return gs
end

# # ----- EvoTrees model -----
# using Flux
# """
#     ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

# The default method to apply the generator loss function to the current counterfactual state for any generator.
# """
# function ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State, s::AbstractArray)

#     loss_fun = !isnothing(generator.loss) ? getfield(Losses, generator.loss) : CounterfactualState.guess_loss(counterfactual_state)
#     @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
#     loss = loss_fun(
#         getfield(Models, :logits)(counterfactual_state.M, counterfactual_state.f(s)), 
#         counterfactual_state.target_encoded
#     )    

#     return loss
# end

# using ForwardDiff
# """
#     ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.EvoTreeModel, counterfactual_state::CounterfactualState.State)

# Method to compute the gradient of the loss function at the current counterfactual state for `EvoTrees.jl` models. It uses `FowardDiff.jl` for gradient access.
# """
# function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.EvoTreeModel, counterfactual_state::CounterfactualState.State)
#     function f(s)
#         ℓ(generator, counterfactual_state, s)
#     end
#     println(ForwardDiff.gradient(f, counterfactual_state.s′))
#     ForwardDiff.gradient(f, counterfactual_state.s′)
# end

# ----- RTorch model -----
using RCall
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual_state::CounterfactualState.State) 
    nn = M.model
    s_cf = counterfactual_state.s′
    t = counterfactual_state.target_encoded
    Interoperability.prep_R_session()
    R"""
    x <- torch_tensor($s_cf, requires_grad=TRUE)
    t <- torch_tensor($t, dtype=torch_float())
    output <- $nn(x)
    obj_loss <- nnf_binary_cross_entropy_with_logits(output,t)
    obj_loss$backward()
    """
    grad = rcopy(R"as_array(x$grad)")
    return grad
end

# ----- PyTorch model -----
using PyCall
"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.RTorchModel, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
function ∂ℓ(generator::AbstractGradientBasedGenerator, M::Models.PyTorchModel, counterfactual_state::CounterfactualState.State) 
    py"""
    import torch
    from torch import nn
    """
    nn = M.model
    s′ = counterfactual_state.s′
    t = counterfactual_state.target_encoded
    x = reshape(s′, 1, length(s′))
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
    ∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State) = gradient(() -> h(generator, counterfactual_state), Flux.params(counterfactual_state.s′))[counterfactual_state.s′]

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators. It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
∇(generator::AbstractGradientBasedGenerator, M::Models.AbstractDifferentiableModel, counterfactual_state::CounterfactualState.State) = ∂ℓ(generator, M, counterfactual_state) + ∂h(generator, counterfactual_state)

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State) 
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state) # gradient
    Δs′ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δs′
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)
    mutability = counterfactual_state.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

"""
    conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
"""
function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState.State)
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state)
    status = all(abs.(𝐠ₜ) .< generator.τ) 
    return status
end

##################################################
# Specific Generators
##################################################

# Baseline
include("GenericGenerator.jl") # Wachter et al. (2017)
include("GreedyGenerator.jl") # Schut et al. (2021)
include("DICEGenerator.jl") # Mothilal et al. (2020)

# Latent space
"""
    AbstractLatentSpaceGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators that search in a latent space. 
"""
abstract type AbstractLatentSpaceGenerator <: AbstractGradientBasedGenerator end

include("REVISEGenerator.jl") # Joshi et al. (2019)
include("CLUEGenerator.jl") # Antoran et al. (2020)