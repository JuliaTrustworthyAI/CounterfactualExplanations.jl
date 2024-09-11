using CounterfactualExplanations
using CounterfactualExplanations.Models
using CounterfactualExplanations.Models: build_mlp
using CounterfactualExplanations.Convergence: GeneratorConditionsConvergence
using Flux
using Flux: Chain
using TaijaData

@testset "MLP" begin
    @testset "Generate counterfactual" begin
        nobs = 1000
        data =
            TaijaData.load_overlapping(nobs) |>
            x -> (Float32.(x[1]), x[2]) |> x -> CounterfactualData(x...)
        M = Models.fit_model(data, :Linear)

        # Select a factual instance:
        target = 2
        factual = 1
        chosen = rand(findall(predict_label(M, data) .== factual))
        x = select_factual(data, chosen)

        # Search:
        conv = GeneratorConditionsConvergence()
        generator = GenericGenerator()
        ce = generate_counterfactual(x, target, data, M, generator; convergence=conv)
        @test typeof(ce) <: CounterfactualExplanation
        @test CounterfactualExplanations.counterfactual_label(ce) == [target]
    end

    @testset "Construct MLP" begin
        build_mlp() |> x -> @test typeof(x) <: Chain
        build_mlp(; dropout=true) |> x -> @test typeof(x) <: Chain
        build_mlp(; batch_norm=true) |> x -> @test typeof(x) <: Chain
    end
end
