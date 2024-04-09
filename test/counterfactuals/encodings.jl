using CounterfactualExplanations.Models: load_mnist_mlp
using MultivariateStats: MultivariateStats
using TaijaData: load_mnist

@testset "encodings.jl" begin
    @testset "Standardize" begin
        dt_standardize = deepcopy(counterfactual_data)
        dt_standardize.standardize = true
        ce = generate_counterfactual(x, target, dt_standardize, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end

    if VERSION >= v"1.8"
        @testset "Dimensionality Reduction" begin
            dt_pca = CounterfactualData(load_mnist(1000)...)
            M = load_mnist_mlp()
            target = 9
            factual = 7
            chosen = rand(findall(predict_label(M, dt_pca) .== factual))
            x = select_factual(dt_pca, chosen)
            dt_pca.dt = MultivariateStats.fit(MultivariateStats.PCA, dt_pca.X; maxoutdim=16)
            generator.dim_reduction = true
            ce = generate_counterfactual(x, target, dt_pca, M, generator)
            @test typeof(ce) <: CounterfactualExplanation
        end
    end
end
