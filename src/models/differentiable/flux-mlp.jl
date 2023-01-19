using Flux
using MLUtils
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
    FluxModelParams

Default MLP training parameters.
"""
@with_kw struct FluxModelParams
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 100
    data_loader::Function = data_loader
end

"""
    train(M::FluxModel, data::CounterfactualData; kwargs...)

Wrapper function to retrain `FluxModel`.
"""
function train(M::FluxModel, data::CounterfactualData; kwargs...)

    args = FluxModelParams(; kwargs...)

    # Prepare data:
    data = args.data_loader(data)

    # Training:
    model = M.model
    forward!(
        model, data;
        loss=args.loss,
        opt=args.opt,
        n_epochs=args.n_epochs
    )

    return M

end

function forward!(model::Flux.Chain, data; loss::Symbol, opt::Symbol, n_epochs::Int=10)

    # Loss:
    loss_(x, y) = getfield(Flux.Losses, loss)(model(x), y)
    avg_loss(data) = mean(map(d -> loss_(d[1], d[2]), data))

    # Optimizer:
    opt_ = getfield(Flux.Optimise, opt)()

    # Training:  
    for epoch = 1:n_epochs
        for d in data
            gs = Flux.gradient(Flux.params(model)) do
                l = loss_(d...)
            end
            Flux.Optimise.update!(opt_, Flux.params(model), gs)
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
function build_mlp(;
    input_dim::Int=2, n_hidden::Int=32, n_layers::Int=2, output_dim::Int=1,
    batch_norm::Bool=false, dropout::Bool=false, activation=Flux.relu,
    p_dropout=0.25
)

    @assert n_layers >= 1 "Need at least one layer."

    if n_layers == 1

        @assert output_dim == 1 "Expected output dimension of 1 for logisitic regression, got $output_dim."

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

function FluxModel(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    input_dim = size(X, 1)
    output_dim = length(unique(y))
    output_dim = output_dim == 2 ? output_dim = 1 : output_dim # adjust in case binary
    model = build_mlp(; input_dim=input_dim, output_dim=output_dim, kwargs...)

    if output_dim == 1
        M = FluxModel(model; likelihood=:classification_binary)
    else
        M = FluxModel(model; likelihood=:classification_multi)
    end

    return M
end


function LogisticRegression(data::CounterfactualData; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    input_dim = size(X, 1)
    output_dim = length(unique(y))
    output_dim = output_dim == 2 ? output_dim = 1 : output_dim # adjust in case binary
    @assert output_dim == 1 "Logistic model not applicable to multi-dimensional output."

    model = build_mlp(; input_dim=input_dim, output_dim=output_dim, n_layers=1, activation=Flux.sigmoid)
    M = FluxModel(model; likelihood=:classification_binary)

    return M
end
