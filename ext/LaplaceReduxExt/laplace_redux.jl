using CounterfactualExplanations.Models

"""
    LaplaceReduxModel <: AbstractCustomDifferentiableModel

Constructor for `LaplaceRedux.jl` model.
"""
struct LaplaceReduxModel <: Models.AbstractCustomDifferentiableModel
    model::LaplaceRedux.Laplace
    likelihood::Symbol
    function LaplaceReduxModel(model, likelihood)
        if likelihood ∈ [:classification_binary, :classification_multi]
            new(model, likelihood)
        else
            throw(
                ArgumentError(
                    "`likelihood` should be in `[:classification_binary, :classification_multi].
                    Support for regressors has not been implemented yet.`",
                ),
            )
        end
        return new(model, likelihood)
    end
end

"""
    CounterfactualExplanations.LaplaceReduxModel(
        model; likelihood::Symbol=:classification_binary
    )

Outer constructor method for `LaplaceReduxModel`.
"""
function CounterfactualExplanations.LaplaceReduxModel(
    model; likelihood::Symbol=:classification_binary
)
    return LaplaceReduxModel(model, likelihood)
end

"""
    logits(M::LaplaceReduxModel, X::AbstractArray)

Predicts the logit scores for the input data `X` using the model `M`.
"""
Models.logits(M::LaplaceReduxModel, X::AbstractArray) =
    LaplaceRedux.predict(M.model, X; predict_proba=false)

"""
    probs(M::LaplaceReduxModel, X::AbstractArray)

Predicts the probabilities of the classes for the input data `X` using the model `M`.
"""
Models.probs(M::LaplaceReduxModel, X::AbstractArray) = LaplaceRedux.predict(M.model, X)

"""
    LaplaceReduxModel(data::CounterfactualData; kwargs...)

Constructs a new LaplaceReduxModel object from the data in a `CounterfactualData` object.
Not called by the user directly.

# Arguments
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `model::LaplaceReduxModel`: The LaplaceRedux model.
"""
function CounterfactualExplanations.LaplaceReduxModel(data::CounterfactualData; kwargs...)
    M_det = FluxModel(data; kwargs...)
    # Laplace wrapper:
    lkli = if M_det.likelihood ∈ [:classification_binary, :classification_multi]
        :classification
    else
        :regression
    end
    la = LaplaceRedux.Laplace(M_det.model; likelihood=lkli)
    M = CounterfactualExplanations.LaplaceReduxModel(la; likelihood=M_det.likelihood)
    return M
end

"""
    train(M::LaplaceReduxModel, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::LaplaceReduxModel`: The wrapper for an LaplaceReduxModel model.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::LaplaceReduxModel`: The fitted LaplaceReduxModel model.
"""
function Models.train(
    M::LaplaceReduxModel,
    data::CounterfactualData;
    train_atomic=true,
    optimize_prior=true,
    kwargs...,
)

    # Setup
    la = M.model

    # Train atomic model
    if train_atomic
        @info "Training atomic model"
        M_atomic = FluxModel(data; kwargs...)
        M_atomic = Models.train(M_atomic, data)
        la = LaplaceRedux.Laplace(M_atomic.model; likelihood=M.model.likelihood)
    end

    # Fit Laplace Approximation
    @info "Fitting Laplace Approximation"
    LaplaceRedux.fit!(la, zip(eachcol(data.X), data.y))

    # Optimize prior
    if optimize_prior
        @info "Optimizing prior"
        LaplaceRedux.optimize_prior!(la)
    end

    return LaplaceReduxModel(la, M.likelihood)
end
