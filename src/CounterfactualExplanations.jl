module CounterfactualExplanations

# Setup:
include("artifacts_setup.jl")

include("base_types.jl")
export AbstractCounterfactualExplanation
export AbstractFittedModel
export AbstractGenerator

# Dependencies:
using Flux

# Global constants:
include("global_utils.jl")
export RawTargetType, EncodedTargetType, RawOutputArrayType, EncodedOutputArrayType
export OutputEncoder
export get_target_index, encode_output

### Data 
# 𝒟 = {(x,y)}ₙ
###
# Generative models for latent space search:
include("generative_models/GenerativeModels.jl")
using .GenerativeModels

# Data preprocessing:
include("data_preprocessing/DataPreprocessing.jl")
using .DataPreprocessing
export CounterfactualData,
    select_factual, apply_domain_constraints, OutputEncoder, transformable_features

### Models 
# ℳ[𝒟] : x ↦ y
###
include("models/Models.jl")
using .Models
export AbstractFittedModel, AbstractDifferentiableModel
export Linear, FluxModel, FluxEnsemble, LaplaceReduxModel, PyTorchModel
export flux_training_params
export probs, logits
export model_catalogue, fit_model, model_evaluation, predict_label

### Objectives
# ℓ( ℳ[𝒟](xᵢ) , target ) + λ cost(xᵢ)
###
include("objectives/Objectives.jl")
using .Objectives

### Generators
# ℓ( ℳ[𝒟](xᵢ) , target )
###
include("generators/Generators.jl")
using .Generators
export AbstractGradientBasedGenerator
export ClaPROARGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export REVISEGenerator
export DiCEGenerator
export WachterGenerator
export generator_catalogue
export generate_perturbations, conditions_satisfied, mutability_constraints
export Generator, @objective, @threshold

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

include("generate_counterfactual.jl")
export generate_counterfactual

include("evaluation/Evaluation.jl")
using .Evaluation

# Precompile:
include("precompile.jl")

end
