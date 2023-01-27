using CategoricalArrays
using Flux
using MLJBase
using Parameters

# Constants:
"""
    RawTargetType

A type union for the allowed types for the `target` variable.
"""
const RawTargetType = Union{Int,AbstractFloat,String,Symbol}

"""
    EncodedTargetType

Type of encoded target variable.
"""
const EncodedTargetType = Real

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
    if stype <: AbstractArray{<:Finite}
        if stype == AbstractVector{Multiclass{2}}
            likelihood = :classification_binary
        else
            likelihood = :classification_multi
        end
    elseif stype <: AbstractArray{Count}
        if length(unique(y)) == 2
            likelihood = :classification_binary
        else
            likelihood = :classification_multi
        end
    elseif stype <: AbstractVector{Continuous}
        error("You supplied an output array of continuous variables, which indicates a regression problem and is not currently supported.")
    else
        error("Could not guess likelihood. Something seems off with your output array.")
    end
    return likelihood, stype
end

"""
    OutputEncoder

The `OutputEncoder` takes a raw output array (`y`) and encodes it.
"""
struct OutputEncoder
    y::RawOutputArrayType
end

"""
    (encoder::OutputEncoder)()

On call, the `OutputEncoder` returns the encoded output array.
"""
function (encoder::OutputEncoder)()

    # Setup:
    y = encoder.y
    likelihood, stype = guess_likelihood(encoder.y)

    # Deal with non-categorical output array:
    if !(stype <: AbstractArray{<:Finite})
        y = categorical(y)
    end

    # Encode:
    y_levels = levels(y)
    y = Int.(y.refs)
    if likelihood == :classification_binary
        y = permutedims(y)
        y = y .- 1  # map to [0,1]
    else
        # One-hot encode:
        y = reduce(
            hcat,
            map(_y -> Flux.onehot(_y[1], 1:length(y_levels)), y)
        )
    end

    return y, y_levels, likelihood

end

"""
    (encoder::OutputEncoder)(ynew::RawTargetType)

When called on a new value `ynew`, the `OutputEncoder` encodes it based on the initial encoding.
"""
function (encoder::OutputEncoder)(ynew::RawTargetType)

    # Setup:
    _, y_levels, likelihood = encoder()
    @assert ynew ∈ y_levels "Supplied output value is not in `y_levels`."

    # Encode:
    y = get_target_index(y_levels, ynew)
    if likelihood == :classification_binary
        y -= 1
    else
        y = Flux.onehot(y, 1:length(y_levels))
    end

    return [y]

end

"""
    get_target_index(y_levels, target)

Utility that returns the index of `target` in `y_levels`.
"""
function get_target_index(y_levels, target)
    @assert target in y_levels "Specified `target` variable does not match any values of `y`."
    findall(y_levels .== target)[1]
end

"""
    FluxModelParams

Default MLP training parameters.
"""
@with_kw mutable struct FluxModelParams
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
        setfield!(flux_training_params, _name, getfield(default_flux_training_params, _name))
    end
    return flux_training_params
end