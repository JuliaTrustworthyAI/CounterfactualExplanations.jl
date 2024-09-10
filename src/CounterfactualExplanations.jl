module CounterfactualExplanations

# Package extensions:
using PackageExtensionCompat: PackageExtensionCompat, @require_extensions
function __init__()
    @require_extensions
end

# Dependencies:
using Flux
using TaijaBase: TaijaBase

# Setup:
include("artifacts_setup.jl")

# Base types:
include("base_types.jl")
export AbstractCounterfactualExplanation
export AbstractModel
export AbstractGenerator
export AbstractConvergence

# Global constants:
include("global_utils.jl")
export RawTargetType, EncodedTargetType, RawOutputArrayType, EncodedOutputArrayType
export OutputEncoder
export get_target_index

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
export AbstractModel
export Linear, MLP, DeepEnsemble
export flux_training_params
export probs, logits
export standard_models_catalogue, all_models_catalogue, model_evaluation, predict_label

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
export AbstractNonGradientBasedGenerator
export ClaPROARGenerator
export ECCoGenerator
export FeatureTweakGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export GrowingSpheresGenerator
export REVISEGenerator
export DiCEGenerator
export WachterGenerator
export generator_catalogue
export generate_perturbations, conditions_satisfied
export @objective

include("convergence/Convergence.jl")
using .Convergence

### CounterfactualExplanation
# argmin 
###
include("counterfactuals/Counterfactuals.jl")
export CounterfactualExplanation
export generate_counterfactual
export total_steps, converged, terminated, path, target_probs
export animate_path

include("evaluation/Evaluation.jl")
using .Evaluation

# Expose necessary functions from extensions:
include("extensions/extensions.jl")

include("deprecated.jl")

end
