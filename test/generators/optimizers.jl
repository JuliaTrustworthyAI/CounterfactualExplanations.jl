using CounterfactualExplanations.Generators: JSMADescent, GreedyGenerator
using TaijaData

# Counteractual data and model:
counterfactual_data = CounterfactualData(TaijaData.load_linearly_separable()...)
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)

@testset "Optimizers" begin
    @testset "JSMADescent" begin
        opt = JSMADescent()
        @test opt.eta == 0.1
        @test opt.n == 10
        n = 5
        opt = JSMADescent(; n=n)
        @test opt.n == n
        @test opt.eta == 1 / n
        @testset "Mutability" begin
            generator = GreedyGenerator()
            mutability = [:both, :none]
            counterfactual_data.mutability = mutability
            ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
            @test isapprox(ce.x[2], x[2]; atol=1e-5)
        end
    end
end
