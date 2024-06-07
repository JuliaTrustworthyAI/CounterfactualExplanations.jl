using Flux: Flux, Chain

"Abstract type for Flux models."
abstract type AbstractFluxNN <: AbstractDifferentiableModelType end

"Concrete type for Flux models."
struct FluxNN <: AbstractFluxNN end

include("utils.jl")
include("Linear.jl")
include("MLP.jl")
include("DeepEnsemble.jl")

"""
    Model(model, type::AbstractFluxNN; likelihood::Symbol=:classification_binary)

Overloaded constructor for Flux models.
"""
function Model(model, type::AbstractFluxNN; likelihood::Symbol=:classification_binary)
    if typeof(model) <: Array
        @.(Flux.testmode!(model))
    else
        Flux.testmode!(model)
    end
    return Model(model, likelihood, model, type)
end

"""
    logits(M::Model, type::AbstractFluxNN, X::AbstractArray)

Overloads the `logits` function for Flux models.
"""
function logits(M::Model, type::AbstractFluxNN, X::AbstractArray)
    return M.fitresult(X)
end

"""
    probs(M::Model, type::AbstractFluxNN, X::AbstractArray)

Overloads the `probs` function for Flux models.
"""
function probs(M::Model, type::AbstractFluxNN, X::AbstractArray)
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
    M::Model, type::AbstractFluxNN, data::CounterfactualData; args=flux_training_params
)

    # Prepare data:
    X, y = (data.X, data.y)
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

    # Store the trained model:
    M.fitresult = Fitresult(model, Dict())

    return M
end
