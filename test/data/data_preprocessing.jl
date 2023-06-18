@testset "Convenience functions" begin
    counterfactual_data = Data.load_overlapping()
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

    counterfactual_data = Data.CounterfactualData(X, y; domain=(0, 0))
    @test unique(DataPreprocessing.apply_domain_constraints(counterfactual_data, x))[1] == 0
end
