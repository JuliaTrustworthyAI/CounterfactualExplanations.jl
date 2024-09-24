using CounterfactualExplanations
using CounterfactualExplanations.Models
using Flux
using MLJBase
using NeuroTreeModels
using TaijaData

@testset "NeuroTreeModel" begin
    # Fit model to data:
    data = CounterfactualData(load_linearly_separable()...)
    M = fit_model(
        data,
        CounterfactualExplanations.NeuroTreeModel();
        depth=2,
        lr=2e-2,
        nrounds=50,
        batchsize=10,
    )

    # Predictions:
    yhat = logits(M, data.X)        # matrix
    yhat = logits(M, data.X[:, 1])  # vector

    # Select a factual instance:
    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    # Generate counterfactual explanation:
    η = 1.0
    generator = GenericGenerator(; opt=Descent(η), λ=0.01)
    conv = CounterfactualExplanations.Convergence.DecisionThresholdConvergence(;
        decision_threshold=0.9, max_iter=250
    )
    ce = generate_counterfactual(x, target, data, M, generator; convergence=conv)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end
