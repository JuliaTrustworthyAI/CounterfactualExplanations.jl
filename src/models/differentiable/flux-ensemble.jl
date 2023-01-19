using Flux
using MLUtils
using SliceMap
using Statistics

"""
    FluxEnsemble <: AbstractDifferentiableJuliaModel

Constructor for deep ensembles trained in `Flux.jl`. 
"""
struct FluxEnsemble <: AbstractDifferentiableJuliaModel
    model::Any
    likelihood::Symbol
    function FluxEnsemble(model, likelihood)
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
function FluxEnsemble(model; likelihood::Symbol=:classification_binary)
    FluxEnsemble(model, likelihood)
end


function logits(M::FluxEnsemble, X::AbstractArray)
    sum(map(nn -> SliceMap.slicemap(x -> nn(x), X, dims=(1, 2)), M.model)) /
    length(M.model)
end

function probs(M::FluxEnsemble, X::AbstractArray)
    if M.likelihood == :classification_binary
        output =
            sum(map(nn -> SliceMap.slicemap(x -> σ.(nn(x)), X, dims=(1, 2)), M.model)) /
            length(M.model)
    elseif M.likelihood == :classification_multi
        output =
            sum(
                map(
                    nn -> SliceMap.slicemap(x -> softmax(nn(x)), X, dims=(1, 2)),
                    M.model,
                ),
            ) / length(M.model)
    end
    return output
end

"""
    FluxModelParams

Default Deep Ensemble training parameters.
"""
@with_kw struct FluxEnsembleParams
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 100
    data_loader::Function = data_loader
end

"""
    train(M::FluxEnsemble, data::CounterfactualData; kwargs...)

Wrapper function to retrain.
"""
function train(M::FluxEnsemble, data::CounterfactualData; kwargs...)

    args = FluxEnsembleParams(; kwargs...)

    # Prepare data:
    data = args.data_loader(data)

    # Training:
    ensemble = M.model

    for model in ensemble
        forward!(
            model, data;
            loss=args.loss,
            opt=args.opt,
            n_epochs=args.n_epochs
        )
    end

    return M

end

"""
    build_ensemble(K::Int;kw=(input_dim=2,n_hidden=32,output_dim=1))

Helper function that builds an ensemble of `K` models.
"""
function build_ensemble(K::Int; kwargs...)
    ensemble = [build_mlp(; kwargs...) for i in 1:K]
    return ensemble
end

function FluxEnsemble(data::CounterfactualData, K::Int=5; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack(data)
    input_dim = size(X, 1)
    output_dim = length(unique(y))
    output_dim = output_dim == 2 ? output_dim = 1 : output_dim # adjust in case binary
    ensemble = build_ensemble(K; input_dim=input_dim, output_dim=output_dim, kwargs...)

    if output_dim == 1
        M = FluxEnsemble(ensemble; likelihood=:classification_binary)
    else
        M = FluxEnsemble(ensemble; likelihood=:classification_multi)
    end

    return M
end
