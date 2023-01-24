using CounterfactualExplanations
using CounterfactualExplanations.Data
import CounterfactualExplanations.DataPreprocessing

@testset "Construction" begin
    for (name, loader) in merge(values(data_catalogue)...)
        @testset "$name" begin
            @test typeof(loader()) == CounterfactualData 
        end
    end
end

@testset "Convenience functions" begin

    counterfactual_data = load_overlapping()
    X = counterfactual_data.X
    y = counterfactual_data.y

    # Select factual:
    idx = rand(1:size(X,2))
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

    counterfactual_data = CounterfactualData(X, y; domain = (0, 0))
    @test unique(apply_domain_constraints(counterfactual_data, x))[1] == 0

end
