using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Models
using Flux
using JointEnergyModels
using MLJFlux
using TaijaBase.Samplers: PCD, SGLD, ImproperSGLD
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
    faith = Evaluation.faithfulness(
        ce; λ=0.1, niter_final=100, n_samples=M.model.batch_size
    )
    plot(ce; zoom=-1)
    X̂ = ce.search[:energy_sampler].posterior
    scatter!(X̂[1, :], X̂[2, :]; label="Posterior Samples", shape=:star5, ms=10)

    # Plot: 
    jem = M.model.jem
    batch_size = M.model.batch_size
    y_labels = Int.(data.output_encoder.labels.refs)
    y = data.output_encoder()[1]
    plts = []
    for target in 1:size(y, 1)
        X̂ = generate_conditional_samples(jem, batch_size, target; niter=1000)
        ex = extrema(hcat(X, X̂); dims=2)
        xlims = ex[1]
        ylims = ex[2]
        x1 = range(1.0f0 .* xlims...; length=100)
        x2 = range(1.0f0 .* ylims...; length=100)
        plt = contour(
            x1,
            x2,
            (x, y) -> softmax(jem([x, y]))[target];
            fill=true,
            alpha=0.5,
            title="Target: $target",
            cbar=true,
            xlims=xlims,
            ylims=ylims,
        )
        scatter!(X[1, :], X[2, :]; color=vec(y_labels), group=vec(y_labels), alpha=0.5)
        scatter!(
            X̂[1, :],
            X̂[2, :];
            color=repeat([target], size(X̂, 2)),
            group=repeat([target], size(X̂, 2)),
            shape=:star5,
            ms=10,
        )
        push!(plts, plt)
    end
    p1 = plot(
        plts...;
        layout=(1, size(y, 1)),
        size=(size(y, 1) * 500, 400),
        plot_title="Pre-trained",
    )

    # From scratch:
    smpler = deepcopy(jem.sampler)
    smpler.buffer = Float32.(rand(smpler.𝒟x, smpler.input_size..., smpler.batch_size))
    rule = jem.sampling_rule
    niter = 250
    PCD(smpler, jem.chain, rule; niter=30, ntransitions=100, y=target)
    plts = []
    for target in 1:size(y, 1)
        X̂ = smpler(jem.chain, rule; y=target, n_samples=batch_size, niter=niter)
        ex = extrema(hcat(X, X̂); dims=2)
        xlims = ex[1]
        ylims = ex[2]
        x1 = range(1.0f0 .* xlims...; length=100)
        x2 = range(1.0f0 .* ylims...; length=100)
        plt = contour(
            x1,
            x2,
            (x, y) -> softmax(jem([x, y]))[target];
            fill=true,
            alpha=0.5,
            title="Target: $target",
            cbar=true,
            xlims=xlims,
            ylims=ylims,
        )
        scatter!(X[1, :], X[2, :]; color=vec(y_labels), group=vec(y_labels), alpha=0.5)
        scatter!(
            X̂[1, :],
            X̂[2, :];
            color=repeat([target], size(X̂, 2)),
            group=repeat([target], size(X̂, 2)),
            shape=:star5,
            ms=10,
        )
        push!(plts, plt)
    end
    p2 = plot(
        plts...;
        layout=(1, size(y, 1)),
        size=(size(y, 1) * 500, 400),
        plot_title="From scratch",
    )
    plot(p1, p2; size=(1000, 600), layout=(2, 1), plot_title="Pre-trained vs. From scratch")

    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
