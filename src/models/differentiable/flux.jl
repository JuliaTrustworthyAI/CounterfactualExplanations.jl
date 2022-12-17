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
function FluxModel(model; likelihood::Symbol = :classification_binary)
    FluxModel(model, likelihood)
end

# Methods
function logits(M::FluxModel, X::AbstractArray)
    return SliceMap.slicemap(x -> M.model(x), X, dims = [1, 2])
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
function FluxEnsemble(model; likelihood::Symbol = :classification_binary)
    FluxEnsemble(model, likelihood)
end


function logits(M::FluxEnsemble, X::AbstractArray)
    sum(map(nn -> SliceMap.slicemap(x -> nn(x), X, dims = [1, 2]), M.model)) /
    length(M.model)
end

function probs(M::FluxEnsemble, X::AbstractArray)
    if M.likelihood == :classification_binary
        output =
            sum(map(nn -> SliceMap.slicemap(x -> σ.(nn(x)), X, dims = [1, 2]), M.model)) /
            length(M.model)
    elseif M.likelihood == :classification_multi
        output =
            sum(
                map(
                    nn -> SliceMap.slicemap(x -> softmax(nn(x)), X, dims = [1, 2]),
                    M.model,
                ),
            ) / length(M.model)
    end
    return output
end
