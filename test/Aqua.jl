using Aqua

@testset "Aqua.jl" begin
    Aqua.test_ambiguities(
        [CounterfactualExplanations];
        recursive=false,
        broken=false
    )

    Aqua.test_all(
        CounterfactualExplanations;
        ambiguities = (recursive=false, broken = false) # I had to add broken=false because otherwise it wasn't a NamedTuple
    )
end
