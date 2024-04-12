@info "Activating local environment"
using Pkg
Pkg.activate(".")

using CounterfactualExplanations
using CounterfactualExplanations.Models
using Flux
using MLJBase
using NeuroTreeModels
using TaijaData

@testset "NeuroTreeModel" begin
    data = CounterfactualData(load_linearly_separable()...)
    M = fit_model(data, :NeuroTree; depth=2, lr=2e-2, nrounds=50)

    target = 2
    factual = 1
    chosen = rand(findall(predict_label(M, data) .== factual))
    x = select_factual(data, chosen)

    η = 1.0
    generator = GenericGenerator(; opt=Descent(η))
    conv = CounterfactualExplanations.Convergence.DecisionThresholdConvergence(;
        decision_threshold=0.9, max_iter=250
    )
    ce = generate_counterfactual(x, target, data, M, generator)
    @test typeof(ce) <: CounterfactualExplanation
    @test CounterfactualExplanations.counterfactual_label(ce) == [target]
end

@info "Deactivating local environment"
Pkg.activate("../../")
