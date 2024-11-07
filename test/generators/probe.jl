using CounterfactualExplanations.Convergence
using CounterfactualExplanations.Objectives

@testset "ProbeGenerator" begin
    @testset "Default arguments" begin
        generator = Generators.ProbeGenerator()
        @test typeof(generator) <: AbstractGenerator
        @test generator.λ == [0.1, 1.0]
        @test generator.loss == Flux.Losses.logitbinarycrossentropy
    end

    @testset "Custom arguments" begin
        generator = Generators.ProbeGenerator(; λ=[0.5, 0.5], loss=:mse)
        @test generator.λ == [0.5, 0.5]
        @test generator.loss == Flux.Losses.mse
    end
end

@testset "hinge_loss" begin
    @testset "Hinge loss calculation" begin
        data = TaijaData.load_linearly_separable()
        counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
            data[1], data[2]
        )
        M = Models.fit_model(counterfactual_data, :Linear)
        target = 2
        factual = 1
        chosen = rand(findall(Models.predict_label(M, counterfactual_data) .== factual))
        x = DataPreprocessing.select_factual(counterfactual_data, chosen)
        # Search:
        generator = Generators.ProbeGenerator()
        linear_counterfactual = CounterfactualExplanations.generate_counterfactual(
            x,
            target,
            counterfactual_data,
            M,
            generator;
            convergence=Convergence.InvalidationRateConvergence(;
                max_iter=100, invalidation_rate=0.1
            ),
        )
        loss = Objectives.hinge_loss(linear_counterfactual)
        rate = Convergence.invalidation_rate(linear_counterfactual)
        @test rate <= 0.1
        @test loss <= 0.9
    end
end
