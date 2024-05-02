"""
    FluxModel <: AbstractFluxModel

Constructor for models trained in `Flux.jl`. 
"""
struct FluxModel <: AbstractFluxModel
    model::Any
    likelihood::Symbol
    function FluxModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            Flux.testmode!(model)
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`type` should be in `[:classification_binary,:classification_multi]`"
                ),
            )
        end
    end
end

# Outer constructor method:
function FluxModel(model; likelihood::Symbol=:classification_binary)
    return FluxModel(model, likelihood)
end

# Methods
function logits(M::FluxModel, X::AbstractArray)
    return M.model(X)
end

function probs(M::FluxModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = Flux.σ.(logits(M, X))
    elseif M.likelihood == :classification_multi
        output = Flux.softmax(logits(M, X))
    end
    return output
end

"""
    train(M::FluxModel, data::CounterfactualData; kwargs...)

Wrapper function to retrain `FluxModel`.
"""
function train(M::FluxModel, data::CounterfactualData; args=flux_training_params)

    # Prepare data:
    data = data_loader(data; batchsize=args.batchsize)

    # Multi-class case:
    if M.likelihood == :classification_multi
        loss = :logitcrossentropy
    else
        loss = args.loss
    end

    # Training:
    model = M.model
    Flux.trainmode!(model)
    forward!(model, data; loss=loss, opt=args.opt, n_epochs=args.n_epochs)
    Flux.testmode!(model)

    return M
end

"""
    FluxModel(data::CounterfactualData; kwargs...)

Constructs a multi-layer perceptron (MLP).
"""
function FluxModel(data::CounterfactualData; kwargs...)

    # Basic setup:
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    input_dim = size(X, 1)
    output_dim = size(y, 1)

    # Build MLP:
    model = build_mlp(; input_dim=input_dim, output_dim=output_dim, kwargs...)

    M = FluxModel(model; likelihood=data.likelihood)

    return M
end

"""
    Linear(data::CounterfactualData; kwargs...)
    
Constructs a model with one linear layer. If the output is binary, this corresponds to logistic regression, since model outputs are passed through the sigmoid function. If the output is multi-class, this corresponds to multinomial logistic regression, since model outputs are passed through the softmax function.
"""
function Linear(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    input_dim = size(X, 1)
    output_dim = size(y, 1)

    model = build_mlp(; input_dim=input_dim, output_dim=output_dim, n_layers=1)

    M = FluxModel(model; likelihood=data.likelihood)

    return M
end
