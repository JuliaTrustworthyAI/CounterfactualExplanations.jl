using Flux
using TaijaData: load_linearly_separable

model = Chain(Dense(20, 2))
data = CounterfactualData(load_linearly_separable()...)

@testset "Deprecations" begin
    @test_deprecated FluxModel(model)
    @test_deprecated FluxModel(data)
    @test_deprecated FluxEnsemble([model, model])
    @test_deprecated FluxEnsemble(data)
end
