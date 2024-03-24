@testset "encodings.jl" begin
    @testset "Standardize" begin
        dt_standardize = deepcopy(counterfactual_data)
        dt_standardize.standardize = true
        ce = generate_counterfactual(x, target, dt_standardize, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end

    @testset "Dimensionality Reduction" begin
        generator.dim_reduction = true
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end
end
