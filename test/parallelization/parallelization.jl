parallelizable(generate_counterfactual)
parallelizable(evaluate)
parallelizable(predict_label)

@testset "Threads" begin
    include("threads.jl")
end
