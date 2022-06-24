using CounterfactualExplanations, CounterfactualExplanations.Interoperability
using Test

@testset "R" begin
    # Installing R deps on remote will results in error:
    @test_throws InteropError Interoperability.prep_R_session()
end
