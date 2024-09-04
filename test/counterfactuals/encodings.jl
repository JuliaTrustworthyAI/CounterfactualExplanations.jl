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

        @testset "SCM generate" begin
            N = 2000
            x = randn(N)
            v = x.*x + randn(N) * 0.25
            w = cos.(x) + randn(N) * 0.25
            z = v + w + randn(N) * 0.25
            s = sin.(z) + randn(N) * 0.25

            df = (x=x, v=v, w=w, z=z, s=s)
            y_lab= Vector{Int}(zeros(2000))
            y_lab .+= rand(0:2,length(y_lab))
            counterfactual_data_scm = CounterfactualData(Tables.matrix(df,transpose=true),y_lab )

            M = fit_model(counterfactual_data_scm, :Linear)
            target = 2
            factual = 1
            chosen = rand(findall(predict_label(M, counterfactual_data_scm) .== factual))
            x = select_factual(counterfactual_data_scm, chosen)
            

            data_scm = deepcopy(counterfactual_data_scm)
            data_scm.input_encoder = fit_transformer(data_scm, CausalInference.SCM)

            generator = GenericGenerator()

            ce = generate_counterfactual(x, target, data_scm, M, generator, initialization=:identity)
            @test typeof(ce) <: CounterfactualExplanation
        end
    end
end
