using CounterfactualExplanations.DataPreprocessing: train_test_split, subsample
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "Convenience functions" begin
    data = TaijaData.load_overlapping()
    counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
        data[1], data[2]
    )
    X = counterfactual_data.X
    y = counterfactual_data.output_encoder.y

    # Select factual:
    idx = Random.rand(1:size(X, 2))
    @test DataPreprocessing.select_factual(counterfactual_data, idx) ==
        counterfactual_data.X[:, idx][:, :]

    # Mutability:
    𝑪 = DataPreprocessing.mutability_constraints(counterfactual_data)
    @test length(𝑪) == size(counterfactual_data.X)[1]
    @test unique(𝑪)[1] == :both

    # Domain:
    x = Random.randn(2)
    @test DataPreprocessing.apply_domain_constraints(counterfactual_data, x) == x

    counterfactual_data = DataPreprocessing.CounterfactualData(X, y; domain=(0, 0))
    @test unique(DataPreprocessing.apply_domain_constraints(counterfactual_data, x))[1] == 0

    @testset "Matrix table" begin
        dataset = Iris()
        X = dataset.features
        y = dataset.targets
        X = MLJBase.table(Tables.matrix(X))
        y = y[:, 1]
        counterfactual_data = CounterfactualData(X, y)
    end
end

@testset "Utilities" begin
    data = TaijaData.load_overlapping()
    counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
        data[1], data[2]
    )
    X = counterfactual_data.X
    y = counterfactual_data.output_encoder.y
    _classes = counterfactual_data.y_levels
    class_ratios = [length(findall(vec(y .== cls))) / length(y) for cls in _classes]

    @testset "Train/test splitting" begin
        test_size = 0.2
        dt_train, dt_test = train_test_split(counterfactual_data; test_size=0.2)
        @test typeof(dt_train) <: CounterfactualData
        @test typeof(dt_test) <: CounterfactualData
        final_test_size = size(dt_test.X, 2) / size(dt_train.X, 2)
        isapprox(final_test_size, test_size; atol=0.1)

        dt_train, dt_test = train_test_split(
            counterfactual_data; test_size=0.2, keep_class_ratio=true
        )
        for _dt in [dt_train, dt_test]
            _y = _dt.output_encoder.y
            _class_ratios = [
                length(findall(vec(_y .== cls))) / length(_y) for cls in _classes
            ]
            for (i, _ratio) in enumerate(_class_ratios)
                @test isapprox(_ratio, class_ratios[i]; atol=1e-5)
            end
        end
    end

    @testset "Random under-/over-sampling" begin
        dt_sub = subsample(counterfactual_data, 10)
        @test size(dt_sub.X, 2) == 10
        dt_sub = subsample(counterfactual_data, Int(1e5))
        @test size(dt_sub.X, 2) == 1e5
    end
end
