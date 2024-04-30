using ChainRulesCore: ignore_derivatives
using MultivariateStats: MultivariateStats
using StatsBase: StatsBase

"""
    encode_array(dt::Nothing, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::Nothing`. This is a no-op.
"""
encode_array(data::CounterfactualData, dt::Nothing, x::AbstractArray) = x

"""
    encode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::MultivariateStats.AbstractDimensionalityReduction`.
"""
function encode_array(
    data::CounterfactualData,
    dt::MultivariateStats.AbstractDimensionalityReduction,
    x::AbstractArray,
)
    return MultivariateStats.predict(dt, x)
end

"""
    encode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::StatsBase.AbstractDataTransform`.
"""
function encode_array(
    data::CounterfactualData, dt::StatsBase.AbstractDataTransform, x::AbstractArray
)
    idx = transformable_features(data)
    ignore_derivatives() do
        s = x[idx, :]
        s = StatsBase.transform(dt, s)
        x[idx, :] = s
    end
    return x
end

"""
    encode_array(dt::GenerativeModels.AbstractGenerativeModel, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::GenerativeModels.AbstractGenerativeModel`.
"""
function encode_array(
    data::CounterfactualData, dt::GenerativeModels.AbstractGenerativeModel, x::AbstractArray
)
    return GenerativeModels.encode(dt, x)
end

"""
    decode_array(dt::Nothing, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::Nothing`. This is a no-op.
"""
decode_array(data::CounterfactualData, dt::Nothing, x::AbstractArray) = x

"""
    decode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::MultivariateStats.AbstractDimensionalityReduction`.
"""
function decode_array(
    data::CounterfactualData,
    dt::MultivariateStats.AbstractDimensionalityReduction,
    x::AbstractArray,
)
    return MultivariateStats.reconstruct(dt, x)
end

"""
    decode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::StatsBase.AbstractDataTransform`.
"""
function decode_array(
    data::CounterfactualData, dt::StatsBase.AbstractDataTransform, x::AbstractArray
)
    idx = transformable_features(data)
    ignore_derivatives() do
        s = x[idx, :]
        s = StatsBase.reconstruct(dt, s)
        x[idx, :] = s
    end
    return x
end

"""
    decode_array(dt::GenerativeModels.AbstractGenerativeModel, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::GenerativeModels.AbstractGenerativeModel`.
"""
function decode_array(
    data::CounterfactualData, dt::GenerativeModels.AbstractGenerativeModel, x::AbstractArray
)
    return GenerativeModels.decode(dt, x)
end

"""
function encode_state(
    ce::CounterfactualExplanation, 
    x::Union{AbstractArray,Nothing} = nothing,
)

Applies all required encodings to `x`:

1. If applicable, it maps `x` to the latent space learned by the generative model.
2. If and where applicable, it rescales features. 

Finally, it returns the encoded state variable.
"""
function encode_state(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)

    # Unpack:
    s′ = isnothing(x) ? deepcopy(ce.s′) : x
    data = ce.data
    dt = data.input_encoder

    # Transform features:
    s′ = encode_array(data, dt, s′)

    return s′
end

"""
    encode_state!(ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing)

In-place version of `encode_state`.
"""
function encode_state!(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)
    ce.s′ = encode_state(ce, x)

    return ce
end

"""
function decode_state(
    ce::CounterfactualExplanation,
    x::Union{AbstractArray,Nothing}=nothing,
)

Applies all the applicable decoding functions:

1. If applicable, map the state variable back from the latent space to the feature space.
2. If and where applicable, inverse-transform features.
3. Reconstruct all categorical encodings.

Finally, the decoded counterfactual is returned.
"""
function decode_state(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)

    # Unpack:
    s′ = isnothing(x) ? deepcopy(ce.s′) : x
    data = ce.data
    dt = data.input_encoder

    # Inverse-transform features:
    s′ = decode_array(data, dt, s′)

    # Categorical:
    s′ = DataPreprocessing.reconstruct_cat_encoding(data, s′)

    return s′
end

"""
    decode_state!(ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing)

In-place version of `decode_state`.
"""
function decode_state!(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)
    ce.x′ = decode_state(ce, x)

    return ce
end
