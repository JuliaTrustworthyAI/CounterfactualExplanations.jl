module CounterfactualExplanations

# Dependencies:
using Flux
import Flux.Losses
using LinearAlgebra

# Interop dependencies:
include("interoperability/Interoperability.jl")
using .Interoperability
export InteropError

### Data 
# 𝒟 = {(x,y)}ₙ
###
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
    FluxModel, FluxEnsemble, LaplaceReduxModel, LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

### Counterfactual state 
# ( ℳ[𝒟] , xᵢ ∈ x )
###

include("counterfactual_state/CounterfactualState.jl")
using .CounterfactualState
export State

### Generators
# ℓ( ℳ[𝒟](xᵢ) , target )
###
include("generators/Generators.jl")
using .Generators
export AbstractGenerator, AbstractGradientBasedGenerator, 
    GenericGenerator, GenericGeneratorParams,
    GreedyGenerator, GreedyGeneratorParams,
    REVISEGenerator, REVISEGeneratorParams,
    DiCEGenerator, DiCEGeneratorParams,
    CLUEGenerator, CLUEGeneratorParams,
    generate_perturbations, conditions_satisified, mutability_constraints   

### CounterfactualExplanation
# argmin 
###

include("counterfactuals/Counterfactuals.jl")
using .Counterfactuals
export CounterfactualExplanation, initialize!, update!,
    total_steps, converged, terminated, path, target_probs

### Other
# Example data sets:
include("data/Data.jl")
using .Data
export load_synthetic, toy_data_linear, toy_data_multi, toy_data_non_linear,
    mnist_data, mnist_ensemble, mnist_model, mnist_vae,
    cats_dogs_data, cats_dogs_model

include("generate_counterfactual.jl")
export generate_counterfactual

include("benchmark/Benchmark.jl")
using .Benchmark

end