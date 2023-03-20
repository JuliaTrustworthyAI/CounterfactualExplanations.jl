using MLUtils
using Plots

generator = generator_catalogue[:generic]()

@testset "Two-dimensional" begin
    M = synthetic[:classification_binary][:models][:MLP][:model]
    counterfactual_data = synthetic[:classification_binary][:data]

    # Model:
    plt = plot(M, counterfactual_data)

    # Counterfactual:
    X = counterfactual_data.X
    x = select_factual(counterfactual_data, rand(1:size(X, 2)))
    y = predict_label(M, counterfactual_data, x)
    target = get_target(counterfactual_data, y[1])
    counterfactual_explanation = generate_counterfactual(
        x, target, counterfactual_data, M, generator
    )
    plt = plot(counterfactual_explanation)
    anim = animate_path(counterfactual_explanation)
end

@testset "Multi-dimensional" begin

    # # Model:
    # xs, ys = (synthetic[:classification_binary][:data][:xs], synthetic[:classification_binary][:data][:ys])
    # X = MLUtils.stack(xs, dims = 2)
    # X = vcat(X, randn(1, size(X, 2)))
    # counterfactual_data = CounterfactualData(X, ys')

    # M = LogisticModel([1 1 0], [0])
    # plt = plot(M, counterfactual_data)

end
