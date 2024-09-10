using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Models
using Flux
using JointEnergyModels
using MLJFlux
using EnergySamplers: PMC, SGLD, ImproperSGLD
using TaijaData

@testset "JEM" begin
    nobs = 1000
    batch_size = Int(round(nobs / 10))
    epochs = 100
    n_hidden = 16
    X, y = TaijaData.load_circles(nobs) |> x -> (Float32.(x[1]), x[2])
    data = CounterfactualData(X, y)
    M = Models.fit_model(
        data,
        :JEM;
        builder=MLJFlux.MLP(; hidden=(n_hidden, n_hidden, n_hidden), σ=Flux.swish),
        batch_size=batch_size,
        finaliser=Flux.softmax,
        loss=Flux.Losses.crossentropy,
        jem_training_params=(α=[1.0, 1.0, 1e-1], verbosity=10),
        epochs=epochs,
        sampling_steps=30,
    )

    # Select a factual instance:
    target = 1
    factual = 0
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    # Search:
    generator = GenericGenerator()
    ce = generate_counterfactual(x, target, data, M, generator)

    # Faithfulness:
    faith = Evaluation.faithfulness(ce; niter_final=100, n_samples=M.model.batch_size)

    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
