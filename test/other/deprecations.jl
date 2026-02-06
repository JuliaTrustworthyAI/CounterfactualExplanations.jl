using Flux
using CounterfactualExplanations.GenerativeModels: VAE
using CounterfactualExplanations.Generators
using TaijaData: load_linearly_separable

model = Chain(Dense(20, 2))
data = CounterfactualData(load_linearly_separable()...)
X = data.X
y = data.y
vae = VAE(size(X, 1))
generator = GenericGenerator()

@testset "Deprecations" begin
    @test_deprecated FluxModel(model)
    @test_deprecated FluxModel(data)
    @test_deprecated FluxEnsemble([model, model])
    @test_deprecated FluxEnsemble(data)
    @test_deprecated CounterfactualExplanations.train!(vae, X, y)
    @test_deprecated CounterfactualExplanations.retrain!(vae, X; n_epochs=10)
    @test_deprecated load_mnist_mlp()
    @test_deprecated load_mnist_ensemble()
end
