using MultivariateStats
using StatsBase

"""
    encode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::MultivariateStats.AbstractDimensionalityReduction`.
"""
encode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray) =
    MultivariateStats.predict(dt, x)

"""
    encode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::StatsBase.AbstractDataTransform`.
"""
encode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray) =
    StatsBase.transform(dt, x)

"""
    decode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::MultivariateStats.AbstractDimensionalityReduction`.
"""
decode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray) =
    MultivariateStats.reconstruct(dt, x)

"""
    decode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::StatsBase.AbstractDataTransform`.
"""
decode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray) =
    StatsBase.reconstruct(dt, x)

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

    # Latent space:
    if ce.params[:latent_space]
        s′ = map_to_latent(ce, s′)
    end

    # Standardize data unless latent space:
    if !ce.params[:latent_space] && data.standardize
        dt = data.dt
        idx = transformable_features(data)
        ChainRulesCore.ignore_derivatives() do
            s = s′[idx, :]
            s = encode_array(dt, s)
            s′[idx, :] = s
        end
    end

    # Compress:
    if data.dt isa MultivariateStats.AbstractDimensionalityReduction &&
        !ce.params[:latent_space] &&
        ce.params[:dim_reduction]
        s′ = encode_array(data.dt, s′)
    end

    return s′
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

    # Latent space:
    if ce.params[:latent_space]
        s′ = map_from_latent(ce, s′)
    end

    # Standardization:
    if !ce.params[:latent_space] && data.standardize
        dt = data.dt

        # Continuous:
        idx = transformable_features(data)
        ChainRulesCore.ignore_derivatives() do
            s = s′[idx, :]
            s = decode_array(dt, s)
            s′[idx, :] = s
        end
    end

    # Decompress:
    if data.dt isa MultivariateStats.AbstractDimensionalityReduction &&
        !ce.params[:latent_space] &&
        ce.params[:dim_reduction]
        s′ = decode_array(data.dt, s′)
    end

    # Categorical:
    s′ = DataPreprocessing.reconstruct_cat_encoding(data, s′)

    return s′
end
