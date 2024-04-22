using CounterfactualExplanations.DataPreprocessing: fit_transformer
using CounterfactualExplanations.Models: load_mnist_mlp
using MultivariateStats: MultivariateStats
using StatsBase: StatsBase
using TaijaData: load_mnist

@testset "encodings.jl" begin
    @testset "Standardize" begin
        dt_standardize = deepcopy(counterfactual_data)
        dt_standardize.input_encoder = fit_transformer(
            dt_standardize, StatsBase.ZScoreTransform
        )
        ce = generate_counterfactual(x, target, dt_standardize, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end

    if VERSION >= v"1.8"
        @testset "Dimensionality Reduction" begin
            dt_pca = CounterfactualData(load_mnist(1000)...)
            dt_pca.input_encoder = fit_transformer(
                dt_pca, MultivariateStats.PCA; maxoutdim=16
            )
            M = load_mnist_mlp()
            target = 9
            factual = 7
            chosen = rand(findall(predict_label(M, dt_pca) .== factual))
            x = select_factual(dt_pca, chosen)
            ce = generate_counterfactual(x, target, dt_pca, M, generator)
            @test typeof(ce) <: CounterfactualExplanation
        end
    end
end
