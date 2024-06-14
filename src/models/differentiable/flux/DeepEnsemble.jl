struct DeepEnsemble <: AbstractFluxNN end

"""
    DeepEnsemble(model; likelihood::Symbol=:classification_binary)

An outer constructor for a deep ensemble model.
"""
function DeepEnsemble(model; likelihood::Symbol=:classification_binary)
    typeof(model) <: Array || error("DeepEnsemble expects an array of models")
    return Model(model, DeepEnsemble(); likelihood=likelihood)
end

"""
    logits(M::Model, type::DeepEnsemble, X::AbstractArray)

Overloads the `logits` function for deep ensembles.
"""
function logits(M::Model, type::DeepEnsemble, X::AbstractArray)
    return sum(map(nn -> nn(X), M.fitresult())) / length(M.fitresult())
end

"""
    probs(M::Model, type::DeepEnsemble, X::AbstractArray)

Overloads the `probs` function for deep ensembles.
"""
function probs(M::Model, type::DeepEnsemble, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = sum(map(nn -> Flux.σ.(nn(X)), M.fitresult())) / length(M.fitresult())
    elseif M.likelihood == :classification_multi
        output = sum(map(nn -> Flux.softmax(nn(X)), M.fitresult())) / length(M.fitresult())
    end
    return output
end

"""
    train(M::Model, type::DeepEnsemble, data::CounterfactualData; kwargs...)

Overloads the `train` function for deep ensembles.
"""
function train(
    M::Model, type::DeepEnsemble, data::CounterfactualData; args=flux_training_params
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
            loss=args.loss,
            opt=args.opt,
            n_epochs=args.n_epochs,
            model_name=msg,
        )

        count += 1
    end

    M.fitresult = Fitresult(ensemble, Dict())

    return M
end

"""
    (M::Model)(data::CounterfactualData, type::DeepEnsemble; kwargs...)
    
Constructs a deep ensemble for the given data.
"""
function (M::Model)(data::CounterfactualData, type::DeepEnsemble; K::Int=5, kwargs...)
    # Basic setup:
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    input_dim = size(X, 1)
    output_dim = size(y, 1)

    # Build deep ensemble:
    ensemble = build_ensemble(K; input_dim=input_dim, output_dim=output_dim, kwargs...)

    M = Model(ensemble, type; likelihood=data.likelihood)

    return M
end
