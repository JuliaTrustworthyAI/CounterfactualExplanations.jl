using Flux
using MLUtils
using ProgressMeter
using SliceMap
using Statistics

"""
    FluxModel <: AbstractDifferentiableJuliaModel

Constructor for models trained in `Flux.jl`. 
"""
struct FluxModel <: AbstractDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function FluxModel(model, likelihood)
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
function FluxModel(model; likelihood::Symbol=:classification_binary)
    FluxModel(model, likelihood)
end

# Methods
function logits(M::FluxModel, X::AbstractArray)
    return SliceMap.slicemap(x -> M.model(x), X, dims=(1, 2))
end

function probs(M::FluxModel, X::AbstractArray)
    if M.likelihood == :classification_binary
        output = σ.(logits(M, X))
    elseif M.likelihood == :classification_multi
        output = softmax(logits(M, X))
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
    forward!(
        model, data;
        loss=loss,
        opt=args.opt,
        n_epochs=args.n_epochs
    )

    return M

end

function forward!(model::Flux.Chain, data; loss::Symbol, opt::Symbol, n_epochs::Int=10, model_name="MLP")

    # Loss:
    loss_(x, y) = getfield(Flux.Losses, loss)(model(x), y)
    avg_loss(data) = mean(map(d -> loss_(d[1], d[2]), data))

    # Optimizer:
    opt_ = getfield(Flux.Optimise, opt)()

    # Training:  
    if flux_training_params.verbose
        @info "Begin training $(model_name)"
        p_epoch = Progress(n_epochs; desc="Progress on epochs:", showspeed=true, color=:green)
    end
    for epoch = 1:n_epochs
        for d in data
            gs = Flux.gradient(Flux.params(model)) do
                l = loss_(d...)
            end
            Flux.Optimise.update!(opt_, Flux.params(model), gs)
        end
        if flux_training_params.verbose
            next!(p_epoch, showvalues=[(:Loss, "$(avg_loss(data))")])
        end
    end

end

"""
    build_mlp()

Helper function to build simple MLP.

# Examples

```julia-repl
nn = build_mlp()
```

"""
function build_mlp(
    ;
    input_dim::Int=2, n_hidden::Int=10, n_layers::Int=2, output_dim::Int=1,
    batch_norm::Bool=false, dropout::Bool=false, activation=Flux.relu,
    p_dropout=0.25
)

    @assert n_layers >= 1 "Need at least one layer."

    if n_layers == 1

        # Logistic regression:
        model = Chain(
            Dense(input_dim, output_dim)
        )

    elseif batch_norm

        hidden_ = repeat([Dense(n_hidden, n_hidden), BatchNorm(n_hidden, activation)], n_layers - 2)

        model = Chain(
            Dense(input_dim, n_hidden),
            BatchNorm(n_hidden, activation),
            hidden_...,
            Dense(n_hidden, output_dim),
            BatchNorm(output_dim)
        )

    elseif dropout

        hidden_ = repeat([Dense(n_hidden, n_hidden, activation), Dropout(p_dropout)], n_layers - 2)

        model = Chain(
            Dense(input_dim, n_hidden, activation),
            Dropout(p_dropout),
            hidden_...,
            Dense(n_hidden, output_dim)
        )
    else

        hidden_ = repeat([Dense(n_hidden, n_hidden, activation)], n_layers - 2)

        model = Chain(
            Dense(input_dim, n_hidden, activation),
            hidden_...,
            Dense(n_hidden, output_dim)
        )

    end

    return model

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
