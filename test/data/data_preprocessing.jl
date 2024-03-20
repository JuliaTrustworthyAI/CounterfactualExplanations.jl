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

@testset "Other" begin
    # MatrixTable:
    dataset = Iris()
    X = dataset.features
    y = dataset.targets
    X = MLJBase.table(Tables.matrix(X))
    y = y[:, 1]
    counterfactual_data = CounterfactualData(X, y)
end