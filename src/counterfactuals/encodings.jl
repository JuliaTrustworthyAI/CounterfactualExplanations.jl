"""
    encode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::MultivariateStats.AbstractDimensionalityReduction`.
"""
function encode_array(
    dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray
)
    return MultivariateStats.predict(dt, x)
end

"""
    encode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::StatsBase.AbstractDataTransform`.
"""
function encode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)
    return StatsBase.transform(dt, x)
end

"""
    decode_array(dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::MultivariateStats.AbstractDimensionalityReduction`.
"""
function decode_array(
    dt::MultivariateStats.AbstractDimensionalityReduction, x::AbstractArray
)
    return MultivariateStats.reconstruct(dt, x)
end

"""
    decode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)

Helper function to decode an array `x` using a data transform `dt::StatsBase.AbstractDataTransform`.
"""
function decode_array(dt::StatsBase.AbstractDataTransform, x::AbstractArray)
    return StatsBase.reconstruct(dt, x)
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

    # Latent space:
    if ce.generator.latent_space
        s′ = map_to_latent(ce, s′)
    end

    # Standardize data unless latent space:
    if !ce.generator.latent_space && data.standardize
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
        !ce.generator.latent_space &&
        ce.generator.dim_reduction
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
    if ce.generator.latent_space
        s′ = map_from_latent(ce, s′)
    end

    # Standardization:
    if !ce.generator.latent_space && data.standardize
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
        !ce.generator.latent_space &&
        ce.generator.dim_reduction
        s′ = decode_array(data.dt, s′)
    end

    # Categorical:
    s′ = DataPreprocessing.reconstruct_cat_encoding(data, s′)

    return s′
end
