using CounterfactualExplanations
using CounterfactualExplanations.Data
import CounterfactualExplanations.DataPreprocessing

@testset "Construction" begin
    xs, ys = Data.toy_data_linear()
    X = hcat(xs...)

    # Default case passes:
    @test typeof(CounterfactualData(X, ys')) == CounterfactualData

    # X not tabular:
    @test_throws MethodError CounterfactualData(xs, ys)

    # Dimension mismatch:
    @test_throws DimensionMismatch typeof(CounterfactualData(X[:, 2:end], ys'))
end

@testset "Convenience functions" begin
    xs, ys = Data.toy_data_linear()
    X = hcat(xs...)
    counterfactual_data = CounterfactualData(X, ys')

    # Select factual:
    idx = rand(1:length(xs))
    @test select_factual(counterfactual_data, idx) == counterfactual_data.X[:, idx][:, :]

    # Mutability:
    𝑪 = CounterfactualExplanations.DataPreprocessing.mutability_constraints(
        counterfactual_data,
    )
    @test length(𝑪) == size(counterfactual_data.X)[1]
    @test unique(𝑪)[1] == :both

    # Domain:
    x = randn(2)
    @test apply_domain_constraints(counterfactual_data, x) == x

    counterfactual_data = CounterfactualData(X, ys'; domain = (0, 0))
    @test unique(apply_domain_constraints(counterfactual_data, x))[1] == 0

end
