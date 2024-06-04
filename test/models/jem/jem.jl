using CounterfactualExplanations
using CounterfactualExplanations.Models
using Flux
using JointEnergyModels
using TaijaData

@testset "JEM" begin
    nobs = 2000
    batch_size = Int(round(nobs / 10))
    data =
        TaijaData.load_circles(nobs) |>
        x -> (Float32.(x[1]), x[2]) |> x -> CounterfactualData(x...)
    M = Models.fit_model(
        data,
        :JEM;
        batch_size=batch_size,
        finaliser=x -> x,
        loss=Flux.Losses.logitcrossentropy,
    )

    # Select a factual instance:
    target = 0
    factual = 1
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    # Search:
    generator = GenericGenerator()
    ce = generate_counterfactual(x, target, data, M, generator)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end