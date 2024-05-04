using Flux: Flux, Chain

"Abstract type for Flux models."
abstract type FluxNN <: AbstractModelType end

"Concrete type for Flux models."
struct FluxNNModel <: FluxNN end

include("utils.jl")
include("Linear.jl")
include("MLP.jl")
include("DeepEnsemble.jl")
include("deprecated/deprecated.jl")

"""
    Model(model, type::FluxNN; likelihood::Symbol=:classification_binary)

Overloaded constructor for Flux models.
"""
function Model(model, type::FluxNN; likelihood::Symbol=:classification_binary)
    if typeof(model) <: Array
        @.(Flux.testmode!(model))
    else
        Flux.testmode!(model)
    end
    return Model(model, likelihood, nothing, type)
end

"""
    logits(M::Model, type::FluxNN, X::AbstractArray)

Overloads the `logits` function for Flux models.
"""
function logits(M::Model, type::FluxNN, X::AbstractArray)
    return M.model(X)
end

"""
    probs(M::Model, type::FluxNN, X::AbstractArray)

Overloads the `probs` function for Flux models.
"""
function probs(M::Model, type::FluxNN, X::AbstractArray)
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
function train(M::Model, type::FluxNN, data::CounterfactualData; args=flux_training_params)

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
