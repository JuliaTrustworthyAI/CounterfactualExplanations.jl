using Aqua

@testset "Aqua.jl" begin
    Aqua.test_all(CounterfactualExplanations;)
end
