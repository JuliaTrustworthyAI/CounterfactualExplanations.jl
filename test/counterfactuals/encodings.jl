using CounterfactualExplanations.DataPreprocessing: fit_transformer
using CounterfactualExplanations.Models: load_mnist_mlp
using CounterfactualExplanations: decode_array
using MultivariateStats: MultivariateStats
using StatsBase: StatsBase
using TaijaData: load_mnist
using Tables
using CausalInference: CausalInference


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
        @testset "Structural Causal Model" begin
            N = 2000
            x = randn(N)
            v = x + randn(N) * 0.25
            w = x + randn(N) * 0.25
            z = v + w + randn(N) * 0.25
            s = z + randn(N) * 0.25

            df = (x=x, v=v, w=w, z=z, s=s)

            data_scm = CounterfactualData(Tables.matrix(df), [0, 1, 1, 2, 1])

            data_scm.input_encoder = fit_transformer(data_scm, CausalInference.SCM)

            x_factual = select_factual(data_scm,1)

            x_decoded= decode_array(
                data_scm,
                data_scm.input_encoder,
                x_factual,
            )

            @test typeof(x_decoded) <: AbstractArray
        end
        @testset "SCM generate" begin
            
            dt_standardize = deepcopy(counterfactual_data)
            dt_standardize.input_encoder = fit_transformer(
                dt_standardize, StatsBase.ZScoreTransform
            )
            ce = generate_counterfactual(x, target, dt_standardize, M, generator)
            @test typeof(ce) <: CounterfactualExplanation
        end
    end
end
