using CounterfactualExplanations
using CounterfactualExplanations.Models
using Flux
using JointEnergyModels
using MLJFlux
using TaijaData

@testset "JEM" begin
    n_obs = 1000
    batch_size = Int(round(nobs / 10))
    epochs = 100
    n_hidden = 16
    data =
        TaijaData.load_blobs(n_obs; cluster_std=0.1, center_box=(-1.0 => 1.0)) |>
        x -> (Float32.(x[1]), x[2]) |> x -> CounterfactualData(x...)
    M = Models.fit_model(
        data,
        :JEM;
        builder=MLJFlux.MLP(; hidden=(n_hidden, n_hidden, n_hidden), σ=Flux.swish),
        batch_size=batch_size,
        finaliser=x -> x,
        loss=Flux.Losses.logitcrossentropy,
        jem_training_params=(α=[1.0, 1.0, 1e-1], verbosity=10),
        epochs=epochs,
        sampling_steps=30,
    )

    # Select a factual instance:
    target = 1
    factual = 2
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    # Search:
    generator = GenericGenerator()
    ce = generate_counterfactual(x, target, data, M, generator)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
