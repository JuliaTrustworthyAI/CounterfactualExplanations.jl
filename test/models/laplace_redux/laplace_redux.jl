using CounterfactualExplanations
using CounterfactualExplanations.Models
using LaplaceRedux
using TaijaData

@testset "LaplaceRedux" begin
    counterfactual_data =
        TaijaData.load_linearly_separable() |>
        x -> (Float32.(x[1]), x[2]) |> x -> CounterfactualData(x...)
    M = Models.fit_model(counterfactual_data, :LaplaceRedux)

    # Select a factual instance:
    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
    x = select_factual(counterfactual_data, chosen)

    # Search:
    generator = GenericGenerator()
    ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
