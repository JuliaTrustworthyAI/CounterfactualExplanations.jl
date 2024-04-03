using CounterfactualExplanations:
    counterfactual_probability_path,
    counterfactual_label_path,
    target_probs_path,
    counterfactual_label

@testset "Path tracking" begin
    ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
    @test typeof(counterfactual_probability_path(ce)) <: Vector
    @test typeof(counterfactual_label_path(ce)) <: Vector
    @test typeof(target_probs_path(ce)) <: Vector
    @test counterfactual_label(ce) == counterfactual_label_path(ce)[end]
end
