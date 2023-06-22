generator = Generators.generator_catalogue[:generic]()

@testset "Two-dimensional" begin
    M = synthetic[:classification_binary][:models][:MLP][:model]
    counterfactual_data = synthetic[:classification_binary][:data]

    # Model:
    plt = Models.plot(M, counterfactual_data)

    # Counterfactual:
    X = counterfactual_data.X
    x = DataPreprocessing.select_factual(counterfactual_data, rand(1:size(X, 2)))
    y = Models.predict_label(M, counterfactual_data, x)
    target = get_target(counterfactual_data, y[1])
    ce = CounterfactualExplanations.generate_counterfactual(
        x, target, counterfactual_data, M, generator
    )
    plt = CounterfactualExplanations.plot(ce)
    anim = CounterfactualExplanations.animate_path(ce)
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
