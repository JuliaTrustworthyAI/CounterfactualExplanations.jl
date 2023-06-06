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
            StatsBase.transform!(dt, s)
            s′[idx, :] = s
        end
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
            StatsBase.reconstruct!(dt, s)
            s′[idx, :] = s
        end
    end

    # Categorical:
    s′ = reconstruct_cat_encoding(ce, s′)

    return s′
end

"""
reconstruct_cat_encoding(
    ce::CounterfactualExplanation,
    x::Union{AbstractArray,Nothing}=nothing,
)

Reconstructs all categorical encodings. See [`DataPreprocessing.reconstruct_cat_encoding`](@ref) for details.
"""
function reconstruct_cat_encoding(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)
    # Unpack:
    s′ = isnothing(x) ? deepcopy(ce.s′) : x
    data = ce.data

    s′ = DataPreprocessing.reconstruct_cat_encoding(data, s′)

    return s′
end