using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.Models
using Flux
using LaplaceRedux
using TaijaData

@testset "LaplaceRedux" begin
    data =
        TaijaData.load_linearly_separable() |>
        x -> (Float32.(x[1]), x[2]) |> x -> CounterfactualData(x...)
    M = Models.fit_model(data, :LaplaceReduxModel)

    # Select a factual instance:
    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    # Search:
    generator = GenericGenerator(; opt=Descent(0.5), λ=0.001)
    conv = MaxIterConvergence(250)
    ce = generate_counterfactual(x, target, data, M, generator; convergence=conv)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
