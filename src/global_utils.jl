using CategoricalArrays: CategoricalArrays, CategoricalArray, CategoricalVector
using Flux: Flux
using MLJBase: MLJBase, Continuous, Count, Finite, Textual, categorical, levels, scitype

# Abstract Base Types:
"""
    RawTargetType

A type union for the allowed types for the `target` variable.
"""
const RawTargetType = Union{Int,AbstractFloat,String,Symbol}

"""
    EncodedTargetType

Type of encoded target variable.
"""
const EncodedTargetType = AbstractArray

"""
    RawOutputArrayType 

A type union for the allowed type for the output array `y`.
"""
const RawOutputArrayType = Union{AbstractVector,AbstractMatrix,CategoricalVector}

"""
    EncodedOutputArrayType

Type of encoded output array.
"""
const EncodedOutputArrayType = AbstractMatrix

"""
    guess_likelihood(y::RawOutputArrayType)

Guess the likelihood based on the scientific type of the output array. Returns a symbol indicating the guessed likelihood and the scientific type of the output array.
"""
function guess_likelihood(y::RawOutputArrayType)
    stype = scitype(y)
    if stype <: Union{AbstractArray{<:Finite},AbstractArray{<:Textual}}
        likelihood = :classification_multi
    elseif stype <: AbstractArray{Count}
        likelihood = :classification_multi
    elseif stype <: AbstractVector{Continuous}
        error(
            "You supplied an output array of continuous variables, which indicates a regression problem and is not currently supported.",
        )
    else
        error("Could not guess likelihood. Something seems off with your output array.")
    end
    return likelihood, stype
end

"""
    OutputEncoder

The `OutputEncoder` takes a raw output array (`y`) and encodes it.
"""
mutable struct OutputEncoder
    y::RawOutputArrayType
    labels::Union{Nothing,CategoricalArray}
end

"""
    (encoder::OutputEncoder)()

On call, the `OutputEncoder` returns the encoded output array.
"""
function (encoder::OutputEncoder)(; return_y::Bool=true)

    # Setup:
    y = encoder.y
    likelihood, stype = guess_likelihood(encoder.y)

    if isnothing(encoder.labels)
        y = ndims(y) == 2 ? vec(y) : y

        # Deal with non-categorical output array:
        if !(stype <: AbstractArray{<:Finite})
            y = categorical(y)
        end
        encoder.labels = y
    end

    # Encode:
    y_levels = levels(y)

    if !return_y
        return y_levels, likelihood
    else
        y = Int.(y.refs)
        if likelihood == :classification_binary
            y = permutedims(y)
            y = y .- 1  # map to [0,1]
        else
            # One-hot encode:
            y = reduce(hcat, map(_y -> Flux.onehot(_y[1], 1:length(y_levels)), y))
        end

        return y, y_levels, likelihood
    end
end

"""
    (encoder::OutputEncoder)(ynew::RawTargetType)

When called on a new value `ynew`, the `OutputEncoder` encodes it based on the initial encoding.
"""
function (encoder::OutputEncoder)(ynew::RawTargetType; y_levels=nothing)

    # Setup:
    if isnothing(y_levels)
        y_levels, likelihood = encoder(; return_y=false)
    else
        likelihood = guess_likelihood(encoder.y)[1]
    end
    @assert ynew ∈ y_levels "Supplied output value is not in `y_levels`."

    # Encode:
    y = get_target_index(y_levels, ynew)
    if likelihood == :classification_binary
        y -= 1
        y = [y]
    else
        y = Flux.onehot(y, 1:length(y_levels))
    end

    return y
end

"""
    get_target_index(y_levels, target)

Utility that returns the index of `target` in `y_levels`.
"""
function get_target_index(y_levels, target)
    @assert target in y_levels "Specified `target` variable does not match any values of `y`."
    return findall(y_levels .== target)[1]
end

"""
    FluxModelParams

Default MLP training parameters.
"""
Base.@kwdef mutable struct FluxModelParams
    loss::Symbol = :logitbinarycrossentropy
    opt::Symbol = :Adam
    n_epochs::Int = 100
    batchsize::Int = 1
    verbose::Bool = false
end

"""
    flux_training_params

The default training parameter for `FluxModels` etc.
"""
const flux_training_params = FluxModelParams()

"""
    reset!(flux_training_params::FluxModelParams)

Restores the default parameter values.
"""
function reset!(flux_training_params::FluxModelParams)
    default_flux_training_params = FluxModelParams()
    for _name in fieldnames(typeof(flux_training_params))
        setfield!(
            flux_training_params, _name, getfield(default_flux_training_params, _name)
        )
    end
    return flux_training_params
end
