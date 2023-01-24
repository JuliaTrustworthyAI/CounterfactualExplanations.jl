module CounterfactualExplanations

abstract type AbstractCounterfactualExplanation end
export AbstractCounterfactualExplanation

# Dependencies:
using Flux
import Flux.Losses

# Global constants:
include("global_utils.jl")
export TargetType, OutputArrayType      # constants
export get_target_index                 # utilities

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
export AbstractFittedModel, AbstractDifferentiableModel
export FluxModel, FluxEnsemble, LaplaceReduxModel, LogisticModel, BayesianLogisticModel
export probs, logits

### Generators
# ℓ( ℳ[𝒟](xᵢ) , target )
###
include("generators/Generators.jl")
using .Generators
export AbstractGenerator, AbstractGradientBasedGenerator
export ClapROARGenerator, ClapROARGeneratorParams
export GenericGenerator, GenericGeneratorParams
export GravitationalGenerator, GravitationalGeneratorParams
export GreedyGenerator, GreedyGeneratorParams
export REVISEGenerator, REVISEGeneratorParams
export DiCEGenerator, DiCEGeneratorParams
export generator_catalog
export generate_perturbations, conditions_satisified, mutability_constraints

### CounterfactualExplanation
# argmin 
###
include("counterfactuals/Counterfactuals.jl")
export CounterfactualExplanation
export initialize!, update!
export total_steps, converged, terminated, path, target_probs
export animate_path


### Other
# Example data sets:
include("data/Data.jl")
using .Data
export load_synthetic, toy_data_linear, toy_data_multi, toy_data_non_linear
export mnist_data, mnist_ensemble, mnist_model, mnist_vae
export cats_dogs_data, cats_dogs_model

include("generate_counterfactual.jl")
export generate_counterfactual

include("benchmark/Benchmark.jl")
using .Benchmark
export benchmark

end
