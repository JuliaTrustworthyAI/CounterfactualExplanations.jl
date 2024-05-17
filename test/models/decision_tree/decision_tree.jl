using CounterfactualExplanations
using CounterfactualExplanations.Models
using DecisionTree
using MLJDecisionTreeInterface
using TaijaData

@testset "DecisionTreeModel" begin
    data =
        TaijaData.load_linearly_separable() |>
        x -> (Float32.(x[1]), x[2]) |> x -> CounterfactualData(x...)
    M = Models.fit_model(data, :RandomForestModel)
    M = Models.fit_model(data, :DecisionTreeModel)

    # Select a factual instance:
    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    # Search:
    generator = FeatureTweakGenerator()
    ce = generate_counterfactual(x, target, data, M, generator)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
