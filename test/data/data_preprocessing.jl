ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "Convenience functions" begin
    data = TaijaData.load_overlapping()
    counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
        data[1], data[2]
    )
    X = counterfactual_data.X
    y = counterfactual_data.output_encoder.y

    # Select factual:
    idx = Random.rand(1:size(X, 2))
    @test DataPreprocessing.select_factual(counterfactual_data, idx) ==
        counterfactual_data.X[:, idx][:, :]

    # Mutability:
    𝑪 = DataPreprocessing.mutability_constraints(counterfactual_data)
    @test length(𝑪) == size(counterfactual_data.X)[1]
    @test unique(𝑪)[1] == :both

    # Domain:
    x = Random.randn(2)
    @test DataPreprocessing.apply_domain_constraints(counterfactual_data, x) == x

    counterfactual_data = DataPreprocessing.CounterfactualData(X, y; domain=(0, 0))
    @test unique(DataPreprocessing.apply_domain_constraints(counterfactual_data, x))[1] == 0
end

@testset "Categorical" begin
    using MLJModels: OneHotEncoder
    y = rand([1, 0], 4)
    X = (
        name=categorical(["Danesh", "Lee", "Mary", "John"]),
        grade=categorical(["A", "B", "A", "C"]; ordered=true),
        sex=categorical(["male", "female", "male", "male"]),
        height=[1.85, 1.67, 1.5, 1.67],
    )
    # Encoding:
    hot = OneHotEncoder()
    mach = MLJBase.fit!(machine(hot, X))
    W = MLJBase.transform(mach, X)
    X = permutedims(MLJBase.matrix(W))
    # Assign:
    features_categorical = [
        [1, 2, 3, 4],      # name
        [5, 6, 7],        # grade
        [8, 9],           # sex
    ]
    features_continuous = [10]
    # Counterfactual data:
    counterfactual_data = CounterfactualData(
        X,
        y;
        features_categorical=features_categorical,
        features_continuous=features_continuous,
    )
end
