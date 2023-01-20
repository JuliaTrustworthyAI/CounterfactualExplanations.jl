@testset "Artifacts" begin

end

@testset "Toy data" begin
    @test length(toy_data_linear()) == 2
    @test length(toy_data_non_linear()) == 2
    @test length(toy_data_multi()) == 2
end
