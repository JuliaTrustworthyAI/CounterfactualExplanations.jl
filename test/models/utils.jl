@testset "Utils" begin
    @testset "Model Evaluation" begin
        counterfactual_data = CounterfactualData(TaijaData.load_linearly_separable()...)
        dt_train, dt_test = CounterfactualExplanations.DataPreprocessing.train_test_split(
            counterfactual_data; test_size=0.7
        )
        M = fit_model(dt_train, :Linear)
        CounterfactualExplanations.Models.model_evaluation(M, dt_test)
        @test true
    end
end
