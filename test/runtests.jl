include("setup.jl")

@testset "CounterfactualExplanations.jl" begin
    include("Aqua.jl")

    @testset "Data" begin
        include("data/data_preprocessing.jl")
    end

    @testset "Generators" begin
        include("generators/generators.jl")
    end

    @testset "Counterfactuals" begin
        include("counterfactuals/counterfactuals.jl")
    end

    @testset "Models" begin
        include("models/models.jl")
    end

    @testset "Evaluation" begin
        include("other/evaluation.jl")
    end

    @testset "Objectives" begin
        include("other/objectives.jl")
    end

    @testset "Other" begin
        include("other/other.jl")
    end

    @testset "Performance" begin
        include("other/performance.jl")
    end

    @testset "Deprecations" begin
        include("other/deprecations.jl")
    end
end
