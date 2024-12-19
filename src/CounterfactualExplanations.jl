module CounterfactualExplanations

# Package extensions:
using PackageExtensionCompat: PackageExtensionCompat, @require_extensions
function __init__()
    @require_extensions
end

# Dependencies:
using Flux: Flux
using TaijaBase: TaijaBase

# Setup:
include("artifacts_setup.jl")

# Base types:
include("base_types.jl")
export AbstractCounterfactualExplanation
export AbstractModel
export AbstractGenerator
export AbstractConvergence
export AbstractPenalty, PenaltyOrFun

# Global constants:
include("global_utils.jl")
export RawTargetType, EncodedTargetType, RawOutputArrayType, EncodedOutputArrayType
export OutputEncoder
export get_target_index

# Error messages:
include("errors.jl")
export NotImplementedModel

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
export fit_model

# Convergence
include("convergence/Convergence.jl")
using .Convergence
export conditions_satisfied

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
export REVISEGenerator
export DiCEGenerator
export WachterGenerator
export generator_catalogue
export generate_perturbations
export @objective

### CounterfactualExplanation
# argmin 
###
include("counterfactuals/Counterfactuals.jl")
export CounterfactualExplanation
export generate_counterfactual
export total_steps, converged, terminated, path, target_probs
export animate_path
export flatten, unflatten, FlattenedCE

include("evaluation/Evaluation.jl")
using .Evaluation
export ExplicitCETransformer, IdentityTransformer, global_ce_transform
export NullSerializer, global_serializer

# Expose necessary functions from extensions:
include("extensions/extensions.jl")

include("deprecated.jl")

end
