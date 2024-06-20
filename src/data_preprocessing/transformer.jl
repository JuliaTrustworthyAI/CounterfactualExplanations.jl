import CausalInference as CI

"""
    fit_transformer!(
        data::CounterfactualData,
        input_encoder::Union{Nothing,InputTransformer,TypedInputTransformer};
        kwargs...,
    )

Fit a transformer to the data in place.
"""
function fit_transformer!(
    data::CounterfactualData,
    input_encoder::Union{Nothing,InputTransformer,TypedInputTransformer};
    kwargs...,
)
    data.input_encoder = fit_transformer(data, input_encoder; kwargs...)
    return data
end

"""
    fit_transformer(data::CounterfactualData, input_encoder::Nothing; kwargs...)

Fit a transformer to the data. This is a no-op if `input_encoder` is `Nothing`.
"""
function fit_transformer(data::CounterfactualData, input_encoder::Nothing; kwargs...)
    return nothing
end

"""
    fit_transformer(data::CounterfactualData, input_encoder::InputTransformer; kwargs...)

Fit a transformer to the data for an `InputTransformer` object. This is a no-op.
"""
function fit_transformer(
    data::CounterfactualData, input_encoder::InputTransformer; kwargs...
)
    return input_encoder
end

"""
    fit_transformer(
        data::CounterfactualData,
        input_encoder::Type{StatsBase.AbstractDataTransform};
        kwargs...,
    )

Fit a transformer to the data for a `StatsBase.AbstractDataTransform` object.
"""
function fit_transformer(
    data::CounterfactualData,
    input_encoder::Type{<:StatsBase.AbstractDataTransform};
    kwargs...,
)
    X = data.X
    dt = StatsBase.fit(
        input_encoder, X[transformable_features(data), :]; dims=ndims(X), kwargs...
    )
    return dt
end

"""
    fit_transformer(
        data::CounterfactualData,
        input_encoder::Type{MultivariateStats.AbstractDimensionalityReduction};
        kwargs...,
    )

Fit a transformer to the data for a `MultivariateStats.AbstractDimensionalityReduction` object.
"""
function fit_transformer(
    data::CounterfactualData,
    input_encoder::Type{<:MultivariateStats.AbstractDimensionalityReduction};
    kwargs...,
)
    X = data.X
    dt = MultivariateStats.fit(input_encoder, X; kwargs...)
    return dt
end

"""
    fit_transformer(
        data::CounterfactualData,
        input_encoder::Type{GenerativeModels.AbstractGenerativeModel};
        kwargs...,
    )

Fit a transformer to the data for a `GenerativeModels.AbstractGenerativeModel` object.
"""
function fit_transformer(
    data::CounterfactualData,
    input_encoder::Type{<:GenerativeModels.AbstractGenerativeModel};
    kwargs...,
)
    X = data.X
    dt = GenerativeModels._fit(input_encoder, X; kwargs...)
    return dt
end


"""
    fit_transformer(
        data::CounterfactualData,
        input_encoder::Type{<:CI.SCM};
        kwargs...,
    )

Fit a transformer to the data for a `SCM` object.
"""
function fit_transformer(
    data::CounterfactualData,
    input_encoder::Type{<:CI.SCM};
    kwargs...,
)
    t = Tables.table(data.X)
    est_g, score = CI.ges(t, penalty=1.0, parallel=true)
    est_dag= CI.pdag2dag!(est_g)
    scm = CI.estimate_equations(t, est_dag)
    return scm
end
