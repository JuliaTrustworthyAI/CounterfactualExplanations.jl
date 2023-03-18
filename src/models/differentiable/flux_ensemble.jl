using Flux
using MLUtils
using Statistics

"""
    FluxEnsemble <: AbstractDifferentiableJuliaModel

Constructor for deep ensembles trained in `Flux.jl`. 
"""
struct FluxEnsemble <: AbstractDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function FluxEnsemble(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi]`",
                ),
            )
        end
    end
end

# Outer constructor method:
function FluxEnsemble(model; likelihood::Symbol = :classification_binary)
    FluxEnsemble(model, likelihood)
end

function logits(M::FluxEnsemble, X::AbstractArray)
    sum(map(nn -> nn(X), M.model)) /
    length(M.model)
end

function probs(M::FluxEnsemble, X::AbstractArray)
    if M.likelihood == :classification_binary
        output =
            sum(map(nn -> σ.(nn(X)), M.model)) /
            length(M.model)
    elseif M.likelihood == :classification_multi
        output =
            sum(map(nn -> softmax(nn(X)), M.model)) / 
            length(M.model)
    end
    return output
end

"""
    FluxModelParams

Default Deep Ensemble training parameters.
"""
@with_kw struct FluxEnsembleParams
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 100
    batchsize::Int = 1
end

"""
    train(M::FluxEnsemble, data::CounterfactualData; kwargs...)

Wrapper function to retrain.
"""
function train(M::FluxEnsemble, data::CounterfactualData; args = flux_training_params)

    # Prepare data:
    data = data_loader(data; batchsize = args.batchsize)

    # Multi-class case:
    if M.likelihood == :classification_multi
        loss = :logitcrossentropy
    else
        loss = args.loss
    end

    # Setup:
    ensemble = M.model
    if flux_training_params.verbose
        @info "Begin training Deep Ensemble"
    end
    count = 1
    n_models = length(ensemble)

    for model in ensemble

        # Model name
        models_done = repeat("#", count)
        models_missing = repeat("-", n_models - count)
        msg = "MLP $(count): $(models_done)$(models_missing) ($(count)/$(n_models))"

        # Train:
        forward!(
            model,
            data;
            loss = args.loss,
            opt = args.opt,
            n_epochs = args.n_epochs,
            model_name = msg,
        )

        count += 1
    end

    return M

end

"""
    build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))

Helper function that builds an ensemble of `K` models.
"""
function build_ensemble(K::Int; kwargs...)
    ensemble = [build_mlp(; kwargs...) for i = 1:K]
    return ensemble
end

function FluxEnsemble(data::CounterfactualData, K::Int = 5; kwargs...)

    # Basic setup:
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    input_dim = size(X, 1)
    output_dim = size(y, 1)

    # Build deep ensemble:
    ensemble = build_ensemble(K; input_dim = input_dim, output_dim = output_dim, kwargs...)

    M = FluxEnsemble(ensemble; likelihood = data.likelihood)

    return M
end
