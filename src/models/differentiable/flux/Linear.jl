struct Linear <: AbstractFluxModelType end

"""
    (M::Model)(data::CounterfactualData, type::LinearClassifier; kwargs...)
    
Constructs a model with one linear layer. If the output is binary, this corresponds to logistic regression, since model outputs are passed through the sigmoid function. If the output is multi-class, this corresponds to multinomial logistic regression, since model outputs are passed through the softmax function.
"""
function (M::Model)(data::CounterfactualData, type::LinearClassifier; kwargs...)
    X, y = CounterfactualExplanations.DataPreprocessing.unpack_data(data)
    input_dim = size(X, 1)
    output_dim = size(y, 1)

    model = build_mlp(; input_dim=input_dim, output_dim=output_dim, n_layers=1)

    M = Model(model, type; likelihood=data.likelihood)

    return M
end
