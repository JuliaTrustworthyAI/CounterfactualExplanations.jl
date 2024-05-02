using Flux: Flux, Chain

abstract type AbstractFluxModelType <: AbstractModelType end

include("utils.jl")
include("Linear.jl")
include("MLP.jl")
include("DeepEnsemble.jl")

function Model(model, type::AbstractFluxModelType; likelihood::Symbol=:classification_binary)
    model = testmode!(model)
    return Model(model, likelihood, nothing, type)
end

# Methods
function logits(M::Model, type::AbstractFluxModelType,  X::AbstractArray)
    return M.model(X)
end

function probs(M::Model, type::AbstractFluxModelType, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = Flux.σ.(logits(M, type, X))
    elseif M.likelihood == :classification_multi
        output = Flux.softmax(logits(M, type, X))
    end
    return output
end

"""
    train(M::FluxModel, data::CounterfactualData; kwargs...)

Wrapper function to train Flux models.
"""
function train(
    M::Model,
    type::AbstractFluxModelType,
    data::CounterfactualData;
    args=flux_training_params,
)

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