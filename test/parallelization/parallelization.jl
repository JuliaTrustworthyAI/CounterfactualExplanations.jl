using CounterfactualExplanations

@test parallelizable(generate_counterfactual)==true
@test parallelizable(CounterfactualExplanations.Evaluation.evaluate)==true
@test parallelizable(predict_label)==false

@testset "Threads" begin
    include("threads.jl")
end
