struct MLP <: AbstractFluxModelType end

"""
    (M::Model)(data::CounterfactualData, type::MLP; kwargs...)
    
Constructs a multi-layer perceptron (MLP).
"""
function (M::Model)(data::CounterfactualData, type::MLP; kwargs...)
    # Basic setup:
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    input_dim = size(X, 1)
    output_dim = size(y, 1)

    # Build MLP:
    model = build_mlp(; input_dim=input_dim, output_dim=output_dim, kwargs...)

    M = Model(model, type; likelihood=data.likelihood)

    return M
end
