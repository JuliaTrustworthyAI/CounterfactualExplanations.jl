using CounterfactualExplanations: output_dim, get_meta, guess_loss

@testset "Utils" begin
    ce = generate_counterfactual(x, factual, counterfactual_data, M, GenericGenerator())
    println(ce)
    ce = generate_counterfactual(x, target, counterfactual_data, M, GenericGenerator())
    println(ce)
    @test typeof(outdim(ce)) <: Int
    @test typeof(get_meta(ce)) <: Dict
end
