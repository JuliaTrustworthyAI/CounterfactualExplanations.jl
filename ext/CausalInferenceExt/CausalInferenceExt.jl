module CausalInferenceExt

using CounterfactualExplanations
using CausalInference

"""
    initialize!(encoder::CausalInference.SCM, ce::AbstractCounterfactualExplanation)

Dispatches the initialization function for an SCM.
"""
function initialize!(encoder::CausalInference.SCM, ce::AbstractCounterfactualExplanation)
    return adjust_shape!(ce) |> encode_state! |> initialize_state! |> decode_state!
end

"""
    run_causal_effects(
        scm::CausalInference.SCM,
        x::AbstractArray,
        idxs::AbstractArray
    )

Apply the causal effects defined in a structural causal model (SCM) to an array `x`.
"""

function run_causal_effects(scm::CausalInference.SCM, x::AbstractArray)
    # Perform the matrix multiplication on the selected rows and include the bias term

    return scm.causal_effects[:, 1:(end - 1)] * x + scm.causal_effects[:, end] # bias

    # try both approaches, split in sum || concatenate 1 in x
end

"""
    decode_array(
        data::CounterfactualData,
        dt::CausalInference.SCM,
        x::AbstractArray,
    )

Helper function to decode an array `x` using a data transform `dt::GenerativeModels.AbstractGenerativeModel`.
"""
function decode_array(data::CounterfactualData, dt::CausalInference.SCM, x::AbstractArray)

    # Apply g(x), as in, either causal parents or identity:
    #x = run_causal_effects(dt, x) # IF no causal parents, THEN identity function, ELSE apply causal effect

    # x₁ = x₁ + u₁
    # x₂ = βx₁ + u₂

    return run_causal_effects(dt, x)
end

"""
    encode_array(data::CounterfactualData, dt::CausalInference.SCM, x::AbstractArray)

Helper function to encode an array `x` using a data transform `dt::CausalInference.SCM`. This is a no-op.
"""
encode_array(data::CounterfactualData, dt::CausalInference.SCM, x::AbstractArray) = x

"""
    transformable_features(
        counterfactual_data::CounterfactualData, input_encoder::Type{CausalInference.SCM}
    )

Returns the indices of all features that have causal parents.
"""
function transformable_features(
    counterfactual_data::CounterfactualData, input_encoder::Type{CausalInference.SCM}
)
    # Find all nodes that have causal parents
    g = counterfactual_data.input_encoder.dag
    child_causal_nodes = [v for v in vertices(g) if indegree(g, v) >= 1]
    return child_causal_nodes
end

"""
    fit_transformer(
        data::CounterfactualData,
        input_encoder::Type{<:CausalInference.SCM};
        kwargs...,
    )

Fit a transformer to the data for a `SCM` object.
"""
function fit_transformer(
    data::CounterfactualData, input_encoder::Type{<:CausalInference.SCM}; kwargs...
)
    t = Tables.table(transpose(data.X))
    est_g, score = CausalInference.ges(t; penalty=1.0, parallel=true)
    est_dag = CausalInference.pdag2dag!(est_g)
    scm = CausalInference.estimate_equations(t, est_dag)
    return scm
end

end
