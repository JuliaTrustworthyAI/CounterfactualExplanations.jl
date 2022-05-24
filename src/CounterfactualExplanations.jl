module CounterfactualExplanations

# Dependencies:
using Flux
import Flux.Losses
using LinearAlgebra

include("data/Data.jl")
using .Data

include("data_preprocessing/DataPreprocessing.jl")
using .DataPreprocessing
export CounterfactualData, select_factual, apply_domain_constraints

include("generative_models/GenerativeModels.jl")
using .GenerativeModels
export vae

include("models/Models.jl")
using .Models
export AbstractFittedModel, AbstractDifferentiableModel, 
    FluxModel, LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

include("generators/Generators.jl")
using .Generators
export AbstractGenerator, AbstractGradientBasedGenerator, GenericGenerator, GreedyGenerator, CounterfactualState,
    generate_perturbations, conditions_satisified, mutability_constraints  

include("counterfactuals/Counterfactuals.jl")
using .Counterfactuals
export CounterfactualExplanation, initialize!, update!,
    total_steps, converged, terminated, path, target_probs

include("generate_counterfactual.jl")
export generate_counterfactual

end