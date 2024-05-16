using CounterfactualExplanations.Models

"""
    CounterfactualExplanations.LaplaceNN(
        model::LaplaceRedux.Laplace; likelihood::Symbol=:classification_binary
    )

Outer constructor for a neural network with Laplace Approximation from `LaplaceRedux.jl`.
"""
function CounterfactualExplanations.LaplaceNN(
    model::LaplaceRedux.Laplace; likelihood::Symbol=:classification_binary
)
    return Models.Model(
        model, CounterfactualExplanations.LaplaceNN(); likelihood=likelihood
    )
end

"""
    (M::Model)(data::CounterfactualData, type::LaplaceNN; kwargs...)
    
Constructs a differentiable tree-based model for the given data.
"""
function (M::Models.Model)(
    data::CounterfactualData, type::CounterfactualExplanations.LaplaceNN; kwargs...
)
    M_det = Models.MLP()(data; kwargs...)
    # Laplace wrapper:
    lkli = if M_det.likelihood ∈ [:classification_binary, :classification_multi]
        :classification
    else
        :regression
    end
    la = LaplaceRedux.Laplace(M_det.model; likelihood=lkli)
    M = CounterfactualExplanations.LaplaceNN(la; likelihood=M_det.likelihood)
    return M
end

"""
    train(M::LaplaceNN, data::CounterfactualData; kwargs...)

Fits the model `M` to the data in the `CounterfactualData` object.
This method is not called by the user directly.

# Arguments
- `M::LaplaceNN`: The wrapper for an LaplaceNN model.
- `data::CounterfactualData`: The `CounterfactualData` object containing the data to be used for training the model.

# Returns
- `M::LaplaceNN`: The fitted LaplaceNN model.
"""
function Models.train(
    M::Models.Model,
    type::CounterfactualExplanations.LaplaceNN,
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
        M_atomic = Models.Model(la.model, Models.FluxNN(); likelihood=M.likelihood)
        M_atomic = Models.train(M_atomic, data)             # standard training for Flux models
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

    # Update model
    M.model = la
    M.fitresult = la

    return M
end

"""
    logits(M::LaplaceNN, X::AbstractArray)

Predicts the logit scores for the input data `X` using the model `M`.
"""
function Models.logits(
    M::Models.Model, type::CounterfactualExplanations.LaplaceNN, X::AbstractArray
)
    return LaplaceRedux.predict(M.fitresult, X; predict_proba=false)
end

"""
    probs(M::LaplaceNN, X::AbstractArray)

Predicts the probabilities of the classes for the input data `X` using the model `M`.
"""
Models.probs(
    M::Models.Model, type::CounterfactualExplanations.LaplaceNN, X::AbstractArray
) = LaplaceRedux.predict(M.fitresult, X)
