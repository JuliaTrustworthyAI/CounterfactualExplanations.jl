using CounterfactualExplanations: output_dim, get_meta

@testset "Utils" begin
    ce = generate_counterfactual(x, target, counterfactual_data, M, GenericGenerator())
    println(ce)
    @test typeof(output_dim(ce)) <: Int
    @test typeof(get_meta(ce)) <: Dict
end
