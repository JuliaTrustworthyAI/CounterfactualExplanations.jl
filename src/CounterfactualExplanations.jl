module CounterfactualExplanations

# Dependencies:
using Flux
import Flux.Losses
using LinearAlgebra

### Data 
# 𝒟 = {(x,y)}ₙ
###

# Example data sets:
include("data/Data.jl")
using .Data

# Generative models for latent space search:
include("generative_models/GenerativeModels.jl")
using .GenerativeModels

# Data preprocessing:
include("data_preprocessing/DataPreprocessing.jl")
using .DataPreprocessing
export CounterfactualData, select_factual, apply_domain_constraints

### Models 
# ℳ[𝒟] : x ↦ y
###

include("models/Models.jl")
using .Models
export AbstractFittedModel, AbstractDifferentiableModel, 
    FluxModel, LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

### Counterfactual state 
# ( ℳ[𝒟] , xᵢ ∈ x )
###

include("counterfactual/CounterfactualState.jl")
using .CounterfactualState

### Generators
# ℓ( ℳ[𝒟](xᵢ) , target )
###
include("generators/Generators.jl")
using .Generators
export AbstractGenerator, AbstractGradientBasedGenerator, GenericGenerator, GreedyGenerator, 
    generate_perturbations, conditions_satisified, mutability_constraints  

### CounterfactualExplanation
# argmin 
###

include("counterfactuals/Counterfactuals.jl")
using .Counterfactuals
export CounterfactualExplanation, initialize!, update!,
    total_steps, converged, terminated, path, target_probs

include("generate_counterfactual.jl")
export generate_counterfactual

end