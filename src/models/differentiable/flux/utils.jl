using ProgressMeter: ProgressMeter
using Statistics: mean

"""
    build_mlp()

Helper function to build simple MLP.

# Examples

```julia-repl
nn = build_mlp()
```

"""
function build_mlp(;
    input_dim::Int=2,
    n_hidden::Int=10,
    n_layers::Int=2,
    output_dim::Int=1,
    dropout::Bool=false,
    batch_norm::Bool=false,
    activation=Flux.relu,
    p_dropout=0.25,
)
    @assert n_layers >= 1 "Need at least one layer."

    if n_layers == 1
        model = Flux.Chain(Flux.Dense(input_dim, output_dim))
    elseif dropout
        hidden_ = repeat(
            [Flux.Dense(n_hidden, n_hidden, activation), Flux.Dropout(p_dropout)],
            n_layers - 2,
        )
        model = Flux.Chain(
            Flux.Dense(input_dim, n_hidden, activation),
            Flux.Dropout(p_dropout),
            hidden_...,
            Flux.Dense(n_hidden, output_dim),
        )
    elseif batch_norm
        hidden_ = repeat(
            [Flux.Dense(n_hidden, n_hidden), Flux.BatchNorm(n_hidden, activation)],
            n_layers - 2,
        )
        model = Chain(
            Flux.Dense(input_dim, n_hidden),
            Flux.BatchNorm(n_hidden, activation),
            hidden_...,
            Flux.Dense(n_hidden, output_dim),
            Flux.BatchNorm(output_dim),
        )
    else
        hidden_ = repeat([Flux.Dense(n_hidden, n_hidden, activation)], n_layers - 2)
        model = Flux.Chain(
            Flux.Dense(input_dim, n_hidden, activation),
            hidden_...,
            Flux.Dense(n_hidden, output_dim),
        )
    end

    return model
end

"""
    forward!(model::Flux.Chain, data; loss::Symbol, opt::Symbol, n_epochs::Int=10, model_name="MLP")

Forward pass for training a `Flux.Chain` model.
"""
function forward!(
    model::Flux.Chain, data; loss::Symbol, opt::Symbol, n_epochs::Int=10, model_name="MLP"
)

    # Loss:
    loss_(x, y) = getfield(Flux.Losses, loss)(x, y)
    avg_loss(data) = mean(map(d -> loss_(model(d[1]), d[2]), data))

    # Optimizer:
    opt_ = getfield(Flux.Optimise, opt)()

    # Training:  
    if flux_training_params.verbose
        @info "Begin training $(model_name)"
        p_epoch = ProgressMeter.Progress(
            n_epochs; desc="Progress on epochs:", showspeed=true, color=:green
        )
    end

    Flux.trainmode!(model)
    opt_state = Flux.setup(opt_, model)
    for epoch in 1:n_epochs
        for d in data
            input, label = d
            gs = Flux.gradient(model) do m
                loss_(m(input), label)
            end
            Flux.Optimise.update!(opt_state, model, gs[1])
        end
        if flux_training_params.verbose
            ProgressMeter.next!(p_epoch; showvalues=[(:Loss, "$(avg_loss(data))")])
        end
    end
    Flux.testmode!(model)

    return model
end

"""
    build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))

Helper function that builds an ensemble of `K` models.
"""
function build_ensemble(K::Int; kwargs...)
    ensemble = [build_mlp(; kwargs...) for i in 1:K]
    return ensemble
end
