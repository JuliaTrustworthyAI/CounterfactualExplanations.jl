include("generative_models.jl")
include("pytorch.jl")

@testset "Standard models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X
            for (likelihood, model) in value[:models]
                name = string(likelihood)
                @testset "$name" begin
                    @testset "Verify correctness of likelihood field" begin
                        @test model[:model].likelihood == value[:data].likelihood
                    end
                    @testset "Matrix of inputs" begin
                        @test size(Models.logits(model[:model], X))[2] == size(X, 2)
                        @test size(Models.probs(model[:model], X))[2] == size(X, 2)
                    end
                    @testset "Vector of inputs" begin
                        @test size(Models.logits(model[:model], X[:, 1]), 2) == 1
                        @test size(Models.probs(model[:model], X[:, 1]), 2) == 1
                    end
                end
            end
        end
    end
end

@testset "Non-standard models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X

            # Test the EvoTree model
            model = Models.fit_model(value[:data], :EvoTree)
            name = "EvoTree"
            @testset "$name" begin
                @testset "Verify correctness of likelihood field" begin
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Matrix of inputs" begin
                    @test size(Models.logits(model, X))[2] == size(X, 2)
                    @test size(Models.probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(Models.logits(model, X[:, 1]), 2) == 1
                    @test size(Models.probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the DecisionTree model
            model = Models.fit_model(value[:data], :DecisionTree)
            name = "DecisionTree"
            @testset "$name" begin
                @testset "Verify correctness of likelihood field" begin
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Matrix of inputs" begin
                    @test size(Models.logits(model, X))[2] == size(X, 2)
                    @test size(Models.probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(Models.logits(model, X[:, 1]), 2) == 1
                    @test size(Models.probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the RandomForest model
            model = Models.fit_model(value[:data], :RandomForest)
            name = "RandomForest"
            @testset "$name" begin
                @testset "Verify correctness of likelihood field" begin
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Matrix of inputs" begin
                    @test size(Models.logits(model, X))[2] == size(X, 2)
                    @test size(Models.probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(Models.logits(model, X[:, 1]), 2) == 1
                    @test size(Models.probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the LaplaceReduxModel
            flux_model = Models.fit_model(value[:data], :Linear).model
            laplace_model = LaplaceRedux.Laplace(flux_model; likelihood=:classification)
            model = Models.LaplaceReduxModel(
                laplace_model; likelihood=:classification_binary
            )

            @testset "Verify correctness of likelihood field for LaplaceRedux" begin
                @test model.likelihood == :classification_binary
            end
        end
    end
end

@testset "Test for errors" begin
    # test Flux models
    @test_throws ArgumentError Models.FluxModel("dummy"; likelihood=:regression)
    @test_throws ArgumentError Models.FluxEnsemble("dummy"; likelihood=:regression)

    data = Data.load_linearly_separable()
    X, y = DataPreprocessing.preprocess_data_for_mlj(data)

    # test the EvoTree model
    M = EvoTrees.EvoTreeClassifier()
    evotree = MLJBase.machine(M, X, y)
    @test_throws ArgumentError Models.EvoTreeModel(evotree; likelihood=:regression)

    # test the DecisionTree model
    M = MLJDecisionTreeInterface.DecisionTreeClassifier()
    tree_model = MLJBase.machine(M, X, y)
    @test_throws ArgumentError Models.TreeModel(tree_model; likelihood=:regression)

    # test the RandomForest model
    M = MLJDecisionTreeInterface.RandomForestClassifier()
    forest_model = MLJBase.machine(M, X, y)
    @test_throws ArgumentError Models.TreeModel(forest_model; likelihood=:regression)

    M = MLJDecisionTreeInterface.DecisionTreeRegressor()
    regression_model = MLJBase.machine(M, X, y)
    @test_throws ArgumentError Models.TreeModel(
        regression_model; likelihood=:classification_binary
    )
    @test_throws ArgumentError Models.TreeModel(
        regression_model; likelihood=:classification_multi
    )

    # test the LaplaceRedux model
    flux_model = Models.fit_model(data, :Linear).model
    laplace_model = LaplaceRedux.Laplace(flux_model; likelihood=:classification)
    @test_throws ArgumentError Models.LaplaceReduxModel(
        laplace_model; likelihood=:classification_multi
    )
    @test_throws ArgumentError Models.LaplaceReduxModel(
        laplace_model; likelihood=:regression
    )
end
